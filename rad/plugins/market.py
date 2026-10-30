"""Market-data plugin: streams L1 book features onto the blackboard while agents run.

This is one example source for the runtime; any async `source(bb)` that publishes works.

Each symbol gets its own coroutine. It replays the reader's 100 ms grid in chunks
(a stand-in for a websocket feed) and carries recursive state across chunks, so
every feature is the exact O(1)-per-tick recursion, evaluated a chunk at a time
with `lfilter` instead of a Python loop:

  vol        fast EW std of 100 ms mid log-returns (~10 s memory), in bps
  vol_x_ref  fast vol / quiet reference vol (from the warm-up history if given, else a ~30 min EW);
             1 = normal, 1.5+ = elevated
  vol_1m_x_ref  realized vol over this chunk (1 min by default) / reference vol
  P_t        Cumulative Pressure Index: P_t = lam * P_{t-1} + z(ofi_t) + z(signed_vol_t),
             where z(.) divides by a running EW std; P_z = P / its own slow EW std
  dP, d2P    first and second time derivatives of P (per second, non-uniform steps)

Per chunk it publishes the freshest snapshot (last-value-wins), a one-line bar to
the log so readers can summarize the path between their reads, and an alert when
vol or the chunk move is abnormal.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..blackboard import Blackboard

TICK_S = 0.1
ALPHA_FAST = 0.01  # ~100 ticks = 10 s
ALPHA_SLOW = 1 / 18000  # ~18k ticks = 30 min
LAM = 0.995  # pressure decay per tick, half-life ~14 s


def _ew(x: np.ndarray, alpha: float, prev: float) -> np.ndarray:
    """y_t = (1 - alpha) * y_{t-1} + alpha * x_t, continuing from y_{-1} = prev."""
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=[(1.0 - alpha) * prev])
    return y


@dataclass
class SymbolState:
    started: bool = False
    last_mid: float = np.nan
    var_ret: float = 0.0
    var_ret_slow: float = 0.0
    var_ref: float = 0.0  # fixed reference variance; 0 = track the slow EW instead
    var_ofi: float = 1.0
    var_sv: float = 1.0
    var_P: float = 1.0
    P: float = 0.0
    t_s: float = 0.0
    P_hist: list[tuple[float, float]] = field(default_factory=list)  # last 3 (t_s, P) chunk-end points


def compute_chunk(st: SymbolState, chunk: pd.DataFrame, sym: str) -> dict:
    bid, ask = chunk[f"{sym}_bid"].to_numpy(), chunk[f"{sym}_ask"].to_numpy()
    ofi, sv = chunk[f"{sym}_ofi"].to_numpy(), chunk[f"{sym}_signed_vol"].to_numpy()
    mid = 0.5 * (bid + ask)

    if not st.started:  # warm-start the EW moments from the first chunk instead of arbitrary constants
        r0 = np.diff(np.log(mid))
        st.var_ret = st.var_ret_slow = float(np.mean(r0**2)) or 1e-12
        st.var_ofi, st.var_sv = float(np.mean(ofi**2)) or 1.0, float(np.mean(sv**2)) or 1.0
        st.last_mid, st.started = mid[0], True

    prev_mid = st.last_mid
    ret = np.diff(np.log(np.concatenate([[prev_mid], mid])))
    var_ret = _ew(ret**2, ALPHA_FAST, st.var_ret)
    var_slow = _ew(ret**2, ALPHA_SLOW, st.var_ret_slow)
    var_ofi = _ew(ofi**2, ALPHA_FAST, st.var_ofi)
    var_sv = _ew(sv**2, ALPHA_FAST, st.var_sv)
    x = ofi / np.sqrt(var_ofi + 1e-12) + sv / np.sqrt(var_sv + 1e-12)
    P, _ = lfilter([1.0], [1.0, -LAM], x, zi=[LAM * st.P])
    var_P = _ew(P**2, ALPHA_SLOW, st.var_P)
    vol_x_ref = np.sqrt(var_ret / ((st.var_ref or var_slow) + 1e-18))

    st.last_mid, st.var_ret, st.var_ret_slow = mid[-1], var_ret[-1], var_slow[-1]
    st.var_ofi, st.var_sv, st.var_P, st.P = var_ofi[-1], var_sv[-1], var_P[-1], P[-1]
    st.t_s += len(chunk) * TICK_S
    st.P_hist = (st.P_hist + [(st.t_s, st.P)])[-3:]
    # Non-uniform finite differences: chunks can differ in length (e.g. a short final chunk).
    h, dP, d2P = st.P_hist, 0.0, 0.0
    if len(h) >= 2:
        dP = (h[-1][1] - h[-2][1]) / (h[-1][0] - h[-2][0])
    if len(h) >= 3:
        dP_prev = (h[-2][1] - h[-3][1]) / (h[-2][0] - h[-3][0])
        d2P = 2 * (dP - dP_prev) / (h[-1][0] - h[-3][0])

    ts = int(chunk.index[-1])
    ticks = len(chunk)
    chunk_ret = 1e4 * np.log(mid[-1] / prev_mid)
    # Abnormal if the chunk move is > 4 sigma of the slow baseline over this many ticks.
    move_z = chunk_ret / (1e4 * np.sqrt((st.var_ref or st.var_ret_slow) * ticks) + 1e-12)
    return {
        "sim_utc": pd.Timestamp(ts, unit="ms").strftime("%H:%M:%S"),
        "ts_ms": ts,
        "ticks": ticks,
        "mid": round(float(mid[-1]), 4),
        "spread_bps": round(float(1e4 * (ask[-1] - bid[-1]) / mid[-1]), 3),
        "chunk_ret_bps": round(float(chunk_ret), 2),
        "chunk_high": round(float(mid.max()), 4),
        "chunk_low": round(float(mid.min()), 4),
        "vol_bps_100ms": round(float(1e4 * np.sqrt(st.var_ret)), 3),
        "vol_x_ref": round(float(vol_x_ref[-1]), 2),
        "vol_x_ref_max": round(float(vol_x_ref.max()), 2),
        "vol_1m_x_ref": round(float(np.sqrt(np.mean(ret**2) / ((st.var_ref or st.var_ret_slow) + 1e-18))), 2),
        "move_z": round(float(move_z), 2),
        "P": round(float(st.P), 2),
        "P_z": round(float(st.P / np.sqrt(st.var_P + 1e-12)), 2),
        "dP_per_s": round(float(dP), 3),
        "d2P_per_s2": round(float(d2P), 4),
    }


async def collect_symbol(bb: Blackboard, grid: pd.DataFrame, sym: str, chunk_ticks: int = 600,
                         speed: float = 300.0, warmup: pd.DataFrame | None = None):
    """Replay `grid` for one symbol; publish features every `chunk_ticks` ticks.

    speed = simulated seconds per wall second (300 -> one simulated minute every 0.2 s).
    warmup = history fed through the recursions first, unpublished, so baselines start filled.
    """
    st = SymbolState()
    if warmup is not None and len(warmup) > 1:
        await asyncio.to_thread(compute_chunk, st, warmup, sym)
        st.P_hist.clear()  # derivatives should only span published chunks
        mid = 0.5 * (warmup[f"{sym}_bid"] + warmup[f"{sym}_ask"]).to_numpy()
        st.var_ref = float(np.mean(np.diff(np.log(mid)) ** 2))
    chunk_wall = chunk_ticks * TICK_S / speed
    for i in range(0, len(grid), chunk_ticks):
        chunk = grid.iloc[i : i + chunk_ticks]
        if len(chunk) < 2:  # a 1-tick tail carries no information
            break
        t0 = time.perf_counter()
        # Feature math runs off the event loop so agent bookkeeping never waits on it.
        f = await asyncio.to_thread(compute_chunk, st, chunk, sym)
        f["compute_us_per_tick"] = round(1e6 * (time.perf_counter() - t0) / len(chunk), 2)
        bb.put_latest(f"market:{sym}", f"collector:{sym}", f)
        bb.append("bar", f"collector:{sym}", {"sym": sym, **f})
        if f["vol_x_ref_max"] >= 3 or abs(f["move_z"]) >= 4:
            bb.append("alert", f"collector:{sym}", {
                "sym": sym, "sim_utc": f["sim_utc"],
                "msg": f"{sym.upper()} {f['chunk_ret_bps']:+.1f} bps in {f['ticks'] / 10:.0f}s "
                       f"(move_z {f['move_z']:+.1f}), vol peak {f['vol_x_ref_max']:.1f}x ref",
            })
        await asyncio.sleep(max(0.0, chunk_wall - (time.perf_counter() - t0)))
    bb.append("source_done", f"collector:{sym}", None)


def summarize_bars(bars: list) -> list[str]:
    """Compress logged bars since the last read into one line per symbol."""
    by_sym: dict[str, list[dict]] = {}
    for e in bars:
        by_sym.setdefault(e.payload["sym"], []).append(e.payload)
    out = []
    for sym, bs in sorted(by_sym.items()):
        first_open = bs[0]["mid"] / np.exp(bs[0]["chunk_ret_bps"] / 1e4)
        ret = 1e4 * np.log(bs[-1]["mid"] / first_open)
        hi, lo = max(b["chunk_high"] for b in bs), min(b["chunk_low"] for b in bs)
        out.append(
            f"- {sym.upper()} {bs[0]['sim_utc']}->{bs[-1]['sim_utc']}: ret {ret:+.1f} bps, "
            f"range {1e4 * np.log(hi / lo):.1f} bps, vol peak {max(b['vol_x_ref_max'] for b in bs):.1f}x ref, "
            f"P_z range [{min(b['P_z'] for b in bs):+.1f}, {max(b['P_z'] for b in bs):+.1f}]"
        )
    return out
