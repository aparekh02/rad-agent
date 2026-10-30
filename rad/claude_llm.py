"""Anthropic SDK backend for swarms.Agent.

swarms.Agent accepts any object with `.run(task) -> str` as its `llm`; with
streaming off it calls exactly that. Plugging the official SDK in here (instead
of swarms' LiteLLM default) gives direct control over thinking, effort, server
tools and refusal handling.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time

import anthropic

MODEL = os.getenv("RAD_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}


class ClaudeLLM:
    def __init__(
        self,
        model: str = MODEL,
        system: str = "",
        effort: str = "low",
        max_tokens: int = 16000,
        web_search: bool = False,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.system = system
        self.effort = effort
        self.max_tokens = max_tokens
        self.tools = [WEB_SEARCH] if web_search else []
        # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
        self.client = client or anthropic.Anthropic(max_retries=3)

    def run(self, task: str | None = None, messages: list[dict] | None = None, **kwargs) -> str:
        # swarms passes its chat history as `messages` (system prompt excluded) and task=None.
        messages = _to_anthropic(messages) if messages else [{"role": "user", "content": task}]
        base = list(messages)
        # Server-side web search can pause a long turn; resend to let it continue.
        for _ in range(4):
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=self.system or anthropic.omit,
                tools=self.tools or anthropic.omit,
                messages=messages,
            )
            if resp.stop_reason == "refusal":
                category = resp.stop_details.category if resp.stop_details else None
                return f"[refused: {category}]"
            if resp.stop_reason != "pause_turn":
                break
            messages = base + [{"role": "assistant", "content": resp.content}]
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def _to_anthropic(messages: list[dict]) -> list[dict]:
    """Keep user/assistant turns, merging consecutive same-role turns (the API wants alternation)."""
    out: list[dict] = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        if m.get("role") == "system":
            continue
        text = str(m.get("content", ""))
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


class ClaudeCodeLLM:
    """Backend that runs headless Claude Code (`claude -p`) on your Claude Code login, no API key.

    Each call is a fresh, tool-less (or web-only) session started from a neutral
    directory, so no project settings, hooks, MCP servers or CLAUDE.md leak in.
    Counts against your Claude plan's usage limits; ~2 s process overhead per call.
    """

    def __init__(self, model: str = MODEL, system: str = "", effort: str = "low", web_search: bool = False, timeout: float = 300, **_):
        self.model, self.system, self.effort, self.timeout = model, system, effort, timeout
        self.tools = "WebSearch,WebFetch" if web_search else ""

    def run(self, task: str | None = None, messages: list[dict] | None = None, **kwargs) -> str:
        prompt = "\n\n".join(m["content"] for m in _to_anthropic(messages)) if messages else task
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model, "--effort", self.effort,
               "--tools", self.tools, "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
        if self.system:
            cmd += ["--system-prompt", self.system]
        if self.tools:
            cmd += ["--allowedTools", self.tools]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=tempfile.gettempdir())
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:300]}")
        if out.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(out.get('result'))[:300]}")
        return str(out.get("result", "")).strip()


class FakeLLM:
    """Offline stand-in with realistic latency so the concurrency is visible without a key."""

    def __init__(self, latency: tuple[float, float] = (0.8, 3.0), reply=None, **_):
        self.latency = latency
        self.reply = reply  # optional callable(task) -> str for consumer-specific canned output

    def run(self, task: str | None = None, messages: list[dict] | None = None, **kwargs) -> str:
        task = task or (messages[-1]["content"] if messages else "")
        time.sleep(random.uniform(*self.latency))
        if self.reply is not None and "RESEARCH QUESTION" not in task:
            return self.reply(task)
        if "RESEARCH QUESTION" in task:
            q = task.split("RESEARCH QUESTION:", 1)[1].strip().splitlines()[0]
            return f"(dry-run finding) Summary for: {q[:80]}"
        return (
            "VIEW: (dry-run) pressure and spreads read from the latest snapshot.\n"
            "RISK: (dry-run) none flagged.\n"
            "STANCE: btc=normal eth=normal sol=flatten\n"
            "RESEARCH: How does order-flow imbalance decay after macro releases?"
        )
