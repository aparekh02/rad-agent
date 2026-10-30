"""Concurrent data collection + processing around running agents.

Batch agent pipelines are phase-barriered: gather everything, wait for the slowest
reader, then synthesize. Here every role runs at the same time on one blackboard:

  sources ──put_latest / append──▶ ┌────────────┐ ◀──append("finding")── research pool (N swarms Agents)
                                   │ Blackboard │                               ▲
  consumer.fast  (every update) ◀──┤            │                               │
  consumer agent (debounced)    ◀──┘            └── follow-up questions ────────┘

- Sources (yours) push processed data: last-value-wins snapshots plus an
  append-only log (bars, alerts, anything).
- `Consumer.fast` is an optional deterministic function run on every update
  (microseconds), so there is always a current decision at data speed.
- The consumer agent wakes on any change, reads the freshest snapshot plus only
  log entries it hasn't seen, and returns text that `Consumer.parse` turns into a
  decision; `Consumer.combine` merges it with the fast path. It never waits on research.
- The research pool answers questions from a queue (seeded by you, extended by the
  agent mid-run via `Consumer.questions`) and posts each finding the moment it's done.
- Agent calls are blocking LLM requests run in threads; the event loop only moves data.

What a decision *means* is entirely up to the consumer.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Callable

os.environ.setdefault("SWARMS_TELEMETRY_ON", "false")  # swarms ships prompts/outputs to its server otherwise

import numpy as np  # noqa: E402
from swarms import Agent  # noqa: E402

from .blackboard import Blackboard, Entry  # noqa: E402

RESEARCH_PROMPT = """You are a research agent feeding a live agent that is processing streaming data.
Answer the question in at most 150 words: key mechanism, any concrete numbers, and source links if you
searched. No preamble."""

INTERNAL_KINDS = {"finding", "decision", "fast_decision", "source_done", "all_done"}


@dataclass
class RoundInput:
    latest: dict[str, Entry]  # freshest snapshot per key
    new: dict[str, list[Entry]]  # source log entries since the last round, by kind
    findings: list[Entry]  # research that is new since the last round
    fast: dict | None  # current fast-path decision
    prev_text: str  # the agent's previous output
    context: str


@dataclass
class Consumer:
    name: str
    system_prompt: str
    render: Callable[[RoundInput], str]
    parse: Callable[[str], dict | None]  # None = reject the output
    fast: Callable[[dict[str, Entry]], dict] | None = None
    combine: Callable[[dict | None, dict], dict] | None = None
    questions: Callable[[str], list[str]] = lambda text: []


@dataclass
class Stats:
    t0: float = field(default_factory=time.monotonic)
    research_s: list[float] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    fast_us: list[float] = field(default_factory=list)
    fast_changes: int = 0
    setup_s: float = 0.0

    def now(self) -> float:
        return time.monotonic() - self.t0

    def report(self, bb: Blackboard) -> str:
        """Pipeline-level numbers: how fresh and how fast the data layer was, independent of use case."""
        log = [e.payload for e in bb.since(0)[0] if isinstance(e.payload, dict)]
        compute = [p["compute_us_per_tick"] for p in log if p.get("compute_us_per_tick") is not None]
        ticks = sum(p.get("ticks", 0) for p in log if "compute_us_per_tick" in p)
        r = self.rounds
        lines = [
            f"agent setup (once, before run) {self.setup_s:.1f}s",
            f"time to first agent decision   {r[0]['at_s']:.1f}s" if r else "no agent decisions",
            f"agent rounds                   {len(r)} (median {np.median([x['llm_s'] for x in r]):.1f}s each)" if r else "",
            f"data age at agent read         median {np.median([x['data_age_s'] for x in r]):.2f}s, "
            f"max {max(x['data_age_s'] for x in r):.2f}s" if r else "",
            f"fast-path evaluations          {len(self.fast_us)} (p50 {np.percentile(self.fast_us, 50):.0f} us, "
            f"p99 {np.percentile(self.fast_us, 99):.0f} us; decision changed {self.fast_changes}x)" if self.fast_us else "",
            f"source snapshots               {sum(bb.snapshot_counts.values())} "
            f"({', '.join(f'{k} {v}' for k, v in sorted(bb.snapshot_counts.items()))})",
            f"ticks processed                {ticks:,} ({ticks / max(self.now(), 1e-9):,.0f}/s wall)" if ticks else "",
            f"processing cost                median {np.median(compute):.2f} us/tick "
            f"(capacity ≈ {1e6 / np.median(compute):,.0f} ticks/s per core; the replay rate above is set by --speed)"
            if compute else "",
            f"research findings              {len(self.research_s)} (median {np.median(self.research_s):.0f}s each, "
            f"run concurrently)" if self.research_s else "",
            f"wall time                      {self.now():.0f}s",
        ]
        return "\n".join(ln for ln in lines if ln)


def make_agent(name: str, system_prompt: str, llm) -> Agent:
    """A bare swarms Agent: one LLM pass per run, the backend (`llm`) holds the system prompt."""
    return Agent(agent_name=name, system_prompt=system_prompt, llm=llm, max_loops=1, print_on=False, output_type="final")


async def research_worker(wid: int, agent: Agent, bb: Blackboard, queue: asyncio.Queue, stats: Stats):
    while True:
        q = await queue.get()
        t = time.monotonic()
        try:
            agent.short_memory.clear()  # each question stands alone
            out = await asyncio.to_thread(agent.run, f"RESEARCH QUESTION: {q}")
            bb.append("finding", f"researcher-{wid}", {"question": q, "answer": str(out)})
        except Exception as e:  # one failed question shouldn't stop the swarm
            bb.append("finding", f"researcher-{wid}", {"question": q, "answer": f"[failed: {e!r}]"})
        finally:
            stats.research_s.append(time.monotonic() - t)
            queue.task_done()


def event_ts(latest: dict[str, Entry]) -> int | None:
    """Newest event time (`ts_ms` in payloads) across snapshots, if sources provide one."""
    ts = [e.payload["ts_ms"] for e in latest.values() if isinstance(e.payload, dict) and "ts_ms" in e.payload]
    return max(ts) if ts else None


class FastPath:
    """Runs `consumer.fast` on every blackboard change and keeps the current decision."""

    def __init__(self, consumer: Consumer, bb: Blackboard, stats: Stats):
        self.consumer, self.bb, self.stats = consumer, bb, stats
        self.current: dict | None = None

    async def run(self):
        seen = -1
        while True:
            await self.bb.wait_change(seen)
            seen = self.bb.version
            latest = self.bb.latest()
            if not latest:
                continue
            t = time.perf_counter()
            d = self.consumer.fast(latest)
            self.stats.fast_us.append(1e6 * (time.perf_counter() - t))
            if d != self.current:
                self.current = d
                self.stats.fast_changes += 1
                self.bb.append("fast_decision", "fast", {"decision": d, "ts_ms": event_ts(latest)})


async def consumer_loop(agent: Agent, bb: Blackboard, queue: asyncio.Queue, consumer: Consumer, stats: Stats,
                        done: asyncio.Event, fast: FastPath | None, min_interval: float = 1.0, max_rounds: int = 40,
                        max_new_questions: int = 4, context: str = "", verbose: bool = True):
    cursor, seen_version, text, asked = 0, -1, "", set()
    for rnd in range(max_rounds):
        if not done.is_set():
            await bb.wait_change(seen_version)
            await asyncio.sleep(min_interval)  # debounce: batch whatever lands meanwhile
        final = done.is_set()
        seen_version = bb.version
        latest = bb.latest()
        if not latest:
            continue
        entries, cursor = bb.since(cursor)
        new: dict[str, list[Entry]] = {}
        for e in entries:
            if e.kind not in INTERNAL_KINDS:
                new.setdefault(e.kind, []).append(e)
        findings = [e for e in entries if e.kind == "finding"]
        if final and not (new or findings) and text:
            return  # nothing new since the last round
        fast_d = fast.current if fast else None
        stale = max(time.monotonic() - e.t for e in latest.values())
        seen_ts = event_ts(latest)
        t = time.monotonic()
        agent.short_memory.clear()  # state is carried explicitly via prev_text, so context stays flat
        prompt = consumer.render(RoundInput(latest, new, findings, fast_d, text, context))
        try:
            out = str(await asyncio.to_thread(agent.run, prompt))
        except Exception as e:  # keep the loop alive; the next round retries with fresher data
            print(f"\n[{stats.now():6.2f}s] round {rnd} agent error: {e!r}")
            continue
        slow = consumer.parse(out)
        if slow is None:
            print(f"\n[{stats.now():6.2f}s] round {rnd}: unparseable output ignored:\n{out[:300]}")
            continue
        text = out
        decision = consumer.combine(fast_d, slow) if consumer.combine else slow
        llm_s = time.monotonic() - t
        bb.append("decision", consumer.name, {"decision": decision, "agent": slow, "fast": fast_d, "ts_ms": seen_ts})
        n_new = sum(len(v) for v in new.values())
        stats.rounds.append({"round": rnd, "at_s": round(stats.now(), 2), "llm_s": round(llm_s, 2),
                             "seen_ts": seen_ts, "arrived_ts": event_ts(bb.latest()),
                             "data_age_s": round(stale, 2), "new_findings": len(findings), "new_entries": n_new,
                             "fast": fast_d, "agent": slow, "final": decision})
        if verbose:
            print(f"\n[{stats.now():6.2f}s] round {rnd} (+{len(findings)} findings, {n_new} new entries, "
                  f"data age ≤{stale:.2f}s)\n{out}\n→ {decision}")
        for q in consumer.questions(out):
            if q not in asked and len(asked) < max_new_questions:
                asked.add(q)
                queue.put_nowait(q)
        if final:
            return


async def run(sources: list, consumer: Consumer, llm_factory, seed_questions: list[str] = (),
              n_researchers: int = 4, **consumer_kw) -> tuple[Stats, Blackboard]:
    """Run sources, the research pool, the fast path and the consumer agent together.

    sources: async callables `source(bb)` that publish onto the blackboard and return when exhausted.
    llm_factory(role, system_prompt): returns an object with `.run(task=..., messages=...) -> str`.
    """
    # swarms.Agent() is synchronous and slow the first time (~1.4 s); build every agent in a thread
    # before any source starts, so construction never stalls the data path.
    t_setup = time.monotonic()

    def build():
        researchers = [make_agent(f"researcher-{i}", RESEARCH_PROMPT, llm_factory("research", RESEARCH_PROMPT))
                       for i in range(n_researchers)]
        main = make_agent(consumer.name, consumer.system_prompt, llm_factory(consumer.name, consumer.system_prompt))
        return researchers, main

    researchers, main_agent = await asyncio.to_thread(build)
    bb, queue, stats, done = Blackboard(), asyncio.Queue(), Stats(), asyncio.Event()
    stats.setup_s = time.monotonic() - t_setup
    for q in seed_questions:
        queue.put_nowait(q)

    fast = FastPath(consumer, bb, stats) if consumer.fast else None
    tasks = [asyncio.create_task(research_worker(i, a, bb, queue, stats)) for i, a in enumerate(researchers)]
    if fast:
        tasks.append(asyncio.create_task(fast.run()))
    feed = [asyncio.create_task(s(bb)) for s in sources]
    agent = asyncio.create_task(consumer_loop(main_agent, bb, queue, consumer, stats, done, fast, **consumer_kw))

    async def finish_when_idle():
        await asyncio.gather(*feed)
        await queue.join()
        done.set()
        bb.append("all_done", "runtime", None)  # wake the agent for its final round

    finisher = asyncio.create_task(finish_when_idle())
    await asyncio.wait([agent, finisher], return_when=asyncio.FIRST_COMPLETED)
    for t in [*tasks, *feed, finisher]:
        t.cancel()
    await agent
    return stats, bb
