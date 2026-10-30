"""Shared state between collectors, researchers and the analyst.

Two kinds of entries, because they need different semantics:
- `put_latest(key, ...)`: last-value-wins (market snapshots). Readers always get
  the freshest state and never work through a backlog of stale ticks.
- `append(kind, ...)`: append-only log (research findings, analyst views). Readers
  keep a cursor and take only what's new since their last read.
Writers wake any waiting reader immediately; nobody polls.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entry:
    kind: str
    source: str
    payload: Any
    t: float = field(default_factory=time.monotonic)


class Blackboard:
    def __init__(self):
        self._log: list[Entry] = []
        self._latest: dict[str, Entry] = {}
        self._version = 0
        self._changed = asyncio.Event()
        self.snapshot_counts: dict[str, int] = {}  # snapshots published per source

    def _bump(self):
        self._version += 1
        self._changed.set()

    def put_latest(self, key: str, source: str, payload: Any):
        self._latest[key] = Entry("latest", source, payload)
        self.snapshot_counts[source] = self.snapshot_counts.get(source, 0) + 1
        self._bump()

    def append(self, kind: str, source: str, payload: Any):
        self._log.append(Entry(kind, source, payload))
        self._bump()

    def latest(self) -> dict[str, Entry]:
        return dict(self._latest)

    def since(self, cursor: int, kinds: set[str] | None = None) -> tuple[list[Entry], int]:
        new = self._log[cursor:]
        if kinds:
            new = [e for e in new if e.kind in kinds]
        return new, len(self._log)

    @property
    def version(self) -> int:
        return self._version

    async def wait_change(self, seen_version: int, timeout: float | None = None) -> bool:
        """Return once anything newer than `seen_version` exists (False on timeout)."""
        while self._version == seen_version:
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout)
            except asyncio.TimeoutError:
                return False
        return True
