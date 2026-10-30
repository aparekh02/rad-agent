"""Quickstart: rad with a synthetic source. Runs with no data files and no API key.

A source streams a noisy sensor reading with occasional spikes. The fast path flags spikes on
every update; the agent summarizes what's happening whenever new data or research lands.

    python examples/quickstart.py                # offline (FakeLLM)
    python examples/quickstart.py --claude-code  # agents on headless Claude Code
    python examples/quickstart.py --live         # agents on the Anthropic API (ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rad import Blackboard, ClaudeCodeLLM, ClaudeLLM, Consumer, FakeLLM, RoundInput, run  # noqa: E402


async def sensor(bb: Blackboard, n: int = 60, period_s: float = 0.1, seed: int = 7):
    """Publish one reading per period: latest snapshot plus a log entry, with a spike now and then."""
    rng = random.Random(seed)
    level, ewma = 20.0, 20.0
    for i in range(n):
        level += rng.gauss(0, 0.2) + (8.0 if rng.random() < 0.05 else 0.0)
        ewma = 0.9 * ewma + 0.1 * level  # O(1) processing on the way in
        reading = {"ts_ms": int(time.time() * 1000), "i": i, "value": round(level, 2), "ewma": round(ewma, 2)}
        bb.put_latest("sensor:temp", "sensor", reading)
        bb.append("reading", "sensor", reading)
        level -= 0.3 * (level - 20.0)  # spikes decay
        await asyncio.sleep(period_s)


def make_consumer() -> Consumer:
    def fast(latest):
        p = latest["sensor:temp"].payload
        return {"spike": abs(p["value"] - p["ewma"]) > 3.0}

    def render(r: RoundInput) -> str:
        vals = [e.payload["value"] for e in r.new.get("reading", [])]
        lines = [f"latest: {r.latest['sensor:temp'].payload}", f"fast path: {r.fast}",
                 f"{len(vals)} new readings: {vals[-20:]}"]
        lines += [f"research: {f.payload['answer']}" for f in r.findings]
        return "\n".join(lines)

    return Consumer(
        name="monitor",
        system_prompt="You monitor a temperature sensor. In one line, say what is happening and whether to alert.",
        render=render,
        parse=lambda text: {"summary": text.splitlines()[0]} if text.strip() else None,
        fast=fast,
        combine=lambda f, a: {**a, **(f or {})},
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-code", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)

    def llm_factory(role, system):
        if a.claude_code:
            return ClaudeCodeLLM(system=system)
        if a.live:
            return ClaudeLLM(system=system)
        return FakeLLM(latency=(0.3, 1.0), reply=lambda task: f"(offline) {task.splitlines()[2]}; no alert")

    stats, bb = asyncio.run(run([sensor], make_consumer(), llm_factory,
                                seed_questions=["What usually causes short spikes in temperature sensor readings?"],
                                n_researchers=2, max_rounds=10, verbose=False))
    for r in stats.rounds:
        print(f"[{r['at_s']:5.1f}s] {r['final']}")
    print("\n" + stats.report(bb))
    return stats


if __name__ == "__main__":
    main()
