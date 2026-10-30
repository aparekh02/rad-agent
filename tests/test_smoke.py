"""Offline smoke tests: no data files, no API key, a few seconds."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from rad import Blackboard  # noqa: E402


def test_blackboard_latest_and_cursor():
    async def go():
        bb = Blackboard()
        bb.put_latest("k", "s", {"v": 1})
        bb.put_latest("k", "s", {"v": 2})  # last value wins
        bb.append("log", "s", 1)
        bb.append("log", "s", 2)
        first, cur = bb.since(0)
        bb.append("log", "s", 3)
        rest, _ = bb.since(cur)  # cursor returns only what's new
        return bb.latest()["k"].payload, [e.payload for e in first], [e.payload for e in rest]

    latest, first, rest = asyncio.run(go())
    assert latest == {"v": 2}
    assert first == [1, 2] and rest == [3]


def test_wait_change_wakes_reader():
    async def go():
        bb = Blackboard()
        v = bb.version
        waiter = asyncio.create_task(bb.wait_change(v, timeout=2))
        await asyncio.sleep(0.01)
        bb.append("x", "s", None)
        return await waiter

    assert asyncio.run(go()) is True


def test_quickstart_runs_concurrently():
    import quickstart

    stats = quickstart.main([])
    assert stats.rounds, "agent made no decisions"
    assert stats.fast_us, "fast path never ran"
    assert all("spike" in r["final"] and "summary" in r["final"] for r in stats.rounds)
    # the agent decided before the source finished (6 s of data): it never waited for the whole feed
    assert stats.rounds[0]["at_s"] < 6.0
