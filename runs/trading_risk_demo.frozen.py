"""Demo consumer: a quoting-risk monitor on Binance BTC/ETH/SOL perpetuals (100 ms L1 grid).

This shows ONE way to consume the runtime's concurrent data layer. What the decision
means (here: a market maker's quoting stance) is the buyer's business; swap in any Consumer.

Data: three market-plugin sources replay 12:00-13:30 UTC as a live feed (11:00-12:00 is
fed first, unpublished, as warm-up and as the quiet reference). Research agents answer
questions in the background.

Decision (fixed before any run; see README "Protocol"):
  fast path  rule, every update: widen if 1-min vol >= 1.5x reference ("vol") or within
             5 min of a scheduled release ("event"); else normal.
  agent      may only ESCALATE to flatten, or STAND DOWN a vol-triggered widen to normal.
  combine    agent flatten -> flatten; rule widen -> widen unless (agent normal and reason vol);
             otherwise normal.

Scoring: each round, per symbol, what happened in the next 5 sim-minutes after the agent's
real (1x) latency: vol vs the quiet reference and max move. Rule, raw agent and combined
policy are scored on the same windows.

    python examples/trading_risk_demo.py --claude-code --web --day 2024-03-11   # train/control day
    python examples/trading_risk_demo.py --claude-code --web --day 2024-03-12   # CPI test day
    python examples/trading_risk_demo.py --dry-run                              # no LLM
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rad import ClaudeCodeLLM, ClaudeLLM, Consumer, FakeLLM, RoundInput, run  # noqa: E402
from rad.plugins.market import collect_symbol, summarize_bars  # noqa: E402

READER = os.path.expanduser(os.getenv("RAD_DATA_DIR", "data"))  # <day>/grid_100ms.pkl
CPI_TS = 1710246600000  # 2024-03-12 12:30:00 UTC
MIN = 60_000
SYMS = ("btc", "eth", "sol")
STANCES = ("normal", "widen", "flatten")

SYSTEM_PROMPT = """You are a real-time risk monitor for a market maker quoting BTC, ETH and SOL perpetuals.
A deterministic rule (the floor) already sets each symbol's stance at data speed:
  widen:vol   = 1-minute vol >= 1.5x the quiet reference
  widen:event = within 5 minutes of a scheduled release
  normal      = neither
You get the floor, the freshest data snapshot, a path summary since your last round, alerts, and new research.

You may change the floor in exactly two ways:
  ESCALATE to flatten: you expect a disorderly next 5 minutes (vol of 3x reference or more, or a large
    one-way move), e.g. a release landing now, |move_z| >= 4 alerts with pressure still building.
  STAND DOWN a widen:vol to normal: the burst is fading. The floor uses 1-minute vol, which lags; if the
    10-second vol_x_ref is already below 1.5 and no alert in this round shows the move still building,
    stand down. With no scheduled event, intraday bursts usually fade within a few minutes.
Otherwise repeat the floor's stance. You cannot stand down widen:event or add a widen the floor did not set.
Standing down when calm earns spread income; flattening when it isn't needed forgoes it.

Field meanings: vol_x_ref / vol_1m_x_ref = vol vs quiet reference (1 = normal). P_z = order-flow pressure in
std units (|P_z| < 2 is noise). move_z = last chunk's move in reference sigmas.

Reply with exactly these lines, no others:
VIEW: <1-2 sentences>
STANCE: btc=<normal|widen|flatten> eth=<normal|widen|flatten> sol=<normal|widen|flatten>
RESEARCH: <one question worth researching in the background, or 'none'>"""

STANCE_RE = re.compile(r"\b(btc|eth|sol)\s*=\s*(normal|widen|flatten)\b", re.I)


def make_consumer(event_ts: int | None) -> Consumer:
    def fast(latest):
        out = {}
        for sym in SYMS:
            e = latest.get(f"market:{sym}")
            if e is None:
                continue
            p = e.payload
            if event_ts is not None and abs(p["ts_ms"] - event_ts) <= 5 * MIN:
                out[sym] = "widen:event"
            elif p["vol_1m_x_ref"] >= 1.5:
                out[sym] = "widen:vol"
            else:
                out[sym] = "normal"
        return out

    def render(r: RoundInput) -> str:
        now = time.monotonic()
        lines = [r.context, "", f"FLOOR (rule, current): {r.fast}", "", "SNAPSHOT (age = wall s since collected):"]
        for key, e in sorted(r.latest.items()):
            p = {k: v for k, v in e.payload.items() if k not in ("ts_ms", "chunk_high", "chunk_low", "compute_us_per_tick")}
            lines.append(f"- {key.split(':')[1].upper()} age={now - e.t:.1f}s {p}")
        if r.new.get("bar"):
            lines += ["", "PATH since your last round:", *summarize_bars(r.new["bar"])]
        if r.new.get("alert"):
            al = r.new["alert"]
            lines += ["", f"ALERTS since your last round ({len(al)}, latest 12):"]
            lines += [f"- {a.payload['sim_utc']} {a.payload['msg']}" for a in al[-12:]]
        if r.findings:
            lines += ["", "NEW RESEARCH:"] + [f"- Q: {f.payload['question']}\n  A: {f.payload['answer']}" for f in r.findings]
        prev = [ln for ln in r.prev_text.splitlines() if ln.upper().startswith(("VIEW:", "STANCE:"))]
        if prev:
            lines += ["", "YOUR PREVIOUS ROUND (continuity only):", *prev]
        return "\n".join(lines)

    def parse(text: str):
        for line in text.splitlines():
            if line.strip().upper().startswith("STANCE:"):
                d = {k.lower(): v.lower() for k, v in STANCE_RE.findall(line)}
                return d if set(d) == set(SYMS) else None
        return None

    def combine(floor: dict | None, agent: dict) -> dict:
        floor = floor or {}
        out = {}
        for sym in SYMS:
            f = floor.get(sym, "normal")
            if agent[sym] == "flatten":
                out[sym] = "flatten"
            elif f.startswith("widen"):
                out[sym] = "normal" if (agent[sym] == "normal" and f == "widen:vol") else "widen"
            else:
                out[sym] = "normal"
        return out

    def questions(text: str):
        for line in text.splitlines():
            if line.upper().startswith("RESEARCH:"):
                q = line.split(":", 1)[1].strip()
                return [q] if q and q.lower() != "none" else []
        return []

    return Consumer("risk-monitor", SYSTEM_PROMPT, render, parse, fast, combine, questions)


def score(grid: pd.DataFrame, rounds: list[dict], ref_window: tuple[int, int], horizon_min: int = 5) -> pd.DataFrame:
    """Per (round, symbol): next-5-min vol vs reference and max move, after the agent's real latency."""
    rows = []
    for sym in SYMS:
        mid = 0.5 * (grid[f"{sym}_bid"] + grid[f"{sym}_ask"])
        lr = np.log(mid).diff()
        ref = lr.loc[ref_window[0] : ref_window[1]].std()
        for r in rounds:
            t = int(r["seen_ts"] + 1000 * r["llm_s"])
            fm = mid.loc[t + 1 : t + horizon_min * MIN]
            rows.append({
                "round": r["round"], "t_utc": pd.Timestamp(t, unit="ms").strftime("%H:%M:%S"), "sym": sym,
                "rule": (r["fast"] or {}).get(sym, "normal").split(":")[0],
                "rule_reason": (r["fast"] or {}).get(sym, "normal"),
                "agent": r["agent"][sym], "final": r["final"][sym],
                "fwd_vol_x_ref": lr.loc[t + 1 : t + horizon_min * MIN].std() / ref,
                "fwd_max_move_bps": 1e4 * np.abs(np.log(fm / mid.loc[:t].iloc[-1])).max(),
            })
    return pd.DataFrame(rows)


def report(sc: pd.DataFrame, risky_x: float = 1.5, severe_x: float = 3.0) -> dict:
    risky, severe = sc["fwd_vol_x_ref"] >= risky_x, sc["fwd_vol_x_ref"] >= severe_x
    print(f"\nwindows: {len(sc)}; risky (next-5-min vol >= {risky_x}x ref): {risky.sum()}; severe (>= {severe_x}x): {severe.sum()}")
    print(f"{'policy':8s} {'recall':>7s} {'precision':>9s} {'quiet-but-cautious':>19s} {'flatten n':>9s} "
          f"{'flatten->severe':>15s} {'normal fwd vol':>14s} {'cautious fwd vol':>16s}")
    out = {}
    for pol in ("rule", "agent", "final"):
        caut = sc[pol] != "normal"
        fl = sc[pol] == "flatten"
        m = {
            "recall": (risky & caut).sum() / max(1, risky.sum()),
            "precision": (risky & caut).sum() / max(1, caut.sum()),
            "quiet_cautious": int((~risky & caut).sum()),
            "flatten_n": int(fl.sum()),
            "flatten_severe": (severe & fl).sum() / max(1, fl.sum()) if fl.any() else float("nan"),
            "normal_fwd_vol": sc.loc[~caut, "fwd_vol_x_ref"].mean(),
            "cautious_fwd_vol": sc.loc[caut, "fwd_vol_x_ref"].mean(),
        }
        out[pol] = m
        print(f"{pol:8s} {m['recall']:7.0%} {m['precision']:9.0%} {m['quiet_cautious']:19d} {m['flatten_n']:9d} "
              f"{m['flatten_severe']:15.0%} {m['normal_fwd_vol']:14.2f} {m['cautious_fwd_vol']:16.2f}")
    sd = sc[(sc["rule_reason"] == "widen:vol") & (sc["final"] == "normal")]
    if len(sd):
        print(f"stand-downs: {len(sd)}, next-5-min vol {sd['fwd_vol_x_ref'].mean():.2f}x ref "
              f"({(sd['fwd_vol_x_ref'] >= risky_x).sum()} turned out risky)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2024-03-11", help="2024-03-11 = train/control (no event); 2024-03-12 = CPI test")
    ap.add_argument("--dry-run", action="store_true", help="FakeLLM; no API key needed")
    ap.add_argument("--claude-code", action="store_true", help="agents run on headless Claude Code (your login)")
    ap.add_argument("--web", action="store_true", help="research agents may use web search")
    ap.add_argument("--speed", type=float, default=20.0, help="simulated seconds per wall second")
    ap.add_argument("--researchers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=45)
    ap.add_argument("--out", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs")))
    a = ap.parse_args()
    if not (a.dry_run or a.claude_code) and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY, or pass --claude-code or --dry-run.")

    grid = pd.read_pickle(f"{READER}/{a.day}/grid_100ms.pkl")
    noon = int(pd.Timestamp(f"{a.day} 12:00", tz="UTC").value // 10**6)
    warm, window = grid.loc[noon - 60 * MIN : noon - 1], grid.loc[noon : noon + 90 * MIN]
    event_ts = CPI_TS if a.day == "2024-03-12" else None
    context = ("SCHEDULED EVENTS (known in advance): " + ("US CPI release at 12:30:00 UTC today." if event_ts else "none today.")
               + " Costs: taker 5 bps, maker 2 bps. Times are UTC.")
    seeds = [
        "How long do intraday volatility bursts in BTC/ETH/SOL perpetuals typically last before vol returns to its prior level?",
        "Around scheduled US macro releases (CPI), how long does elevated crypto volatility persist after the print?",
        "Does order-flow imbalance predict the next few minutes of volatility in crypto perpetuals?",
        "What signals distinguish an orderly repricing from a disorderly liquidation cascade in crypto perps?",
    ]

    def llm_factory(role: str, system: str):
        if a.dry_run:
            return FakeLLM(latency=(2.0, 6.0) if role == "research" else (0.5, 1.5))
        backend = ClaudeCodeLLM if a.claude_code else ClaudeLLM
        return backend(system=system, effort="low", web_search=a.web and role == "research")

    sources = [functools.partial(collect_symbol, grid=window, sym=s, speed=a.speed, warmup=warm) for s in SYMS]
    print(f"{a.day}: replaying {len(window):,} ticks x {len(SYMS)} symbols at {a.speed:.0f}x "
          f"≈ {len(window) / 10 / a.speed:.0f}s wall; event: {'CPI 12:30' if event_ts else 'none'}")
    stats, bb = asyncio.run(run(sources, make_consumer(event_ts), llm_factory, seeds,
                                n_researchers=a.researchers, max_rounds=a.rounds, context=context))

    print("\n=== data layer ===")
    print(stats.report(bb))
    if not stats.rounds:
        return
    sc = score(grid, stats.rounds, (noon - 60 * MIN, noon))
    print("\n=== demo consumer: quoting stance vs what the market did next ===")
    metrics = report(sc)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.day}_{time.strftime('%H%M%S')}{'_dry' if a.dry_run else ''}"
    sc.to_csv(f"{a.out}/{tag}_scores.csv", index=False)
    with open(f"{a.out}/{tag}_rounds.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in stats.rounds)
    with open(f"{a.out}/{tag}_metrics.json", "w") as f:
        json.dump({"day": a.day, "speed": a.speed, "pipeline": stats.report(bb), "policies": metrics}, f, indent=1, default=float)
    print(f"\nsaved {a.out}/{tag}_*")


if __name__ == "__main__":
    main()
