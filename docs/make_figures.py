"""Render the README results figure (light + dark) from a saved demo run.

    .venv/bin/python docs/make_figures.py runs/2024-03-12_114008_rounds.jsonl
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

READER = os.path.expanduser(os.getenv("RAD_DATA_DIR", "data"))  # <day>/grid_100ms.{parquet,pkl}


def load_grid(day: str) -> pd.DataFrame:
    """Load a day's 100 ms grid. Prefers Parquet; a pickle is only safe if you created it yourself."""
    base = f"{READER}/{day}/grid_100ms"
    if os.path.exists(base + ".parquet"):
        return pd.read_parquet(base + ".parquet")
    return pd.read_pickle(base + ".pkl")  # pickle runs code on load: never use one from an untrusted source
CPI_TS = 1710246600000
MIN = 60_000
OUT = os.path.dirname(os.path.abspath(__file__))

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e1e0d9",
                  axis="#c3c2b7", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e86", grid="#2c2c2a",
                 axis="#383835", series="#3987e5"),
}
# Stances are states, so they use the fixed status palette, always with a text label.
STATUS = {"normal": "#0ca30c", "widen": "#fab219", "flatten": "#d03b3b"}


def load(rounds_path: str):
    rounds = [json.loads(line) for line in open(rounds_path)]
    grid = load_grid("2024-03-12")
    noon = CPI_TS - 30 * MIN
    g = grid.loc[noon - 60 * MIN : noon + 95 * MIN]
    mid = 0.5 * (g["btc_bid"] + g["btc_ask"])
    lr = np.log(mid).diff()
    ref = lr.loc[noon - 60 * MIN : noon].std()
    # Realized vol over the NEXT 5 minutes at each second, vs the quiet reference.
    sec = lr.loc[noon : noon + 90 * MIN]
    fwd = (sec[::-1].rolling(3000, min_periods=3000).std()[::-1] / ref).iloc[::10]
    return rounds, mid.loc[noon : noon + 90 * MIN], fwd, noon


def intervals(rounds, key, end_ts):
    """(start, end, stance) per round for BTC; each stance acts from when the agent's answer arrives (1x)."""
    starts = [int(r["seen_ts"] + 1000 * r["llm_s"]) for r in rounds]
    out = []
    for i, r in enumerate(rounds):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(rounds) else end_ts
        val = (r[key] or {}).get("btc", "normal").split(":")[0]
        out.append((s, e, val))
    return out


def ts(x):
    return pd.to_datetime(x, unit="ms")


def render(rounds, mid, fwd, noon, mode: str):
    t = THEMES[mode]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": t["text"],
                         "axes.labelcolor": t["text2"], "xtick.color": t["text2"], "ytick.color": t["text2"]})
    fig = plt.figure(figsize=(11, 7.2), facecolor=t["surface"])
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 0.9, 1.6, 0.05], hspace=0.35, left=0.11, right=0.97, top=0.83, bottom=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(length=0)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(ts(CPI_TS), color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlim(ts(noon), ts(noon + 90 * MIN))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))

    fig.text(0.11, 0.95, "One CPI release, processed live", fontsize=16, fontweight="bold", color=t["text"])
    fig.text(0.11, 0.91, "BTC perpetual, 2024-03-12 12:00–13:30 UTC, replayed as a live feed. Every decision below acts only "
                         "after the agent's real latency.", fontsize=10, color=t["text2"])
    fig.text(0.11, 0.875, "Untuned test day: the prompt was frozen on a different (quiet) day before this run.",
             fontsize=10, color=t["text2"])

    ax = axes[0]
    ax.plot(ts(mid.index[::10]), mid.values[::10], color=t["series"], linewidth=2)
    ax.set_ylabel("BTC mid (USD)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.text(ts(CPI_TS + 20_000), ax.get_ylim()[1], "  US CPI 12:30", va="top", color=t["text2"], fontsize=9)

    ax = axes[1]
    end = noon + 90 * MIN
    rows = [("Rule + agent", "final"), ("Rule only", "fast")]  # listed top to bottom
    for y, (label, key) in enumerate(rows[::-1]):
        for s, e, val in intervals(rounds, key, end):
            ax.barh(y, (ts(e) - ts(s)), left=ts(s), height=0.62, color=STATUS[val], edgecolor=t["surface"], linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], color=t["text"])
    ax.grid(False)
    ax.set_ylim(-0.45, 1.75)
    ax.legend(handles=[Patch(color=STATUS[k], label=lbl) for k, lbl in
                       (("normal", "normal: quote as usual"), ("widen", "widen: wider, smaller"),
                        ("flatten", "flatten: pull quotes"))],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=t["text"],
              handlelength=1.2)
    flat = [r for r in rounds if (r["final"] or {}).get("btc") == "flatten"]
    if flat:
        x = ts(int(flat[0]["seen_ts"] + 1000 * flat[0]["llm_s"]))
        ax.annotate("agent escalates to flatten at the print", xy=(x + pd.Timedelta(minutes=1), 1.32),
                    xytext=(ts(CPI_TS + 8 * MIN), 1.62),
                    color=t["text"], fontsize=9, arrowprops=dict(arrowstyle="-", color=t["text2"], linewidth=1))

    ax = axes[2]
    ax.plot(ts(fwd.index), fwd.values, color=t["series"], linewidth=2)
    ax.axhline(1.5, color=t["text2"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ts(noon + 60_000), 1.5, "risky ≥ 1.5×", va="bottom", color=t["text2"], fontsize=9)
    ax.set_ylabel("next-5-min vol\n× quiet level")
    ax.set_ylim(0, max(6.5, float(np.nanmax(fwd.values)) * 1.08))
    ax.set_xlabel("UTC")

    for a in axes[:2]:
        a.tick_params(labelbottom=False)
    path = f"{OUT}/results-{mode}.png"
    fig.savefig(path, dpi=160, facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = load(sys.argv[1] if len(sys.argv) > 1 else "runs/2024-03-12_114008_rounds.jsonl")
    for m in THEMES:
        print(render(*data, m))
