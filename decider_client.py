"""Minimal synchronous client for a running Strands Decider server.

Adapted from examples/strands/_client.py in strands-labs/strands-decider (Apache-2.0).
Only the standard library is used, so the decision pattern can be dropped into any project.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

DEFAULT_URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")


class DeciderUnavailable(RuntimeError):
    """Raised when no decider server answers at the configured URL."""


class Decider:
    """One state, many typed questions, one round trip."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.last_latency_ms: float | None = None
        self.last_server_latency_ms: float | None = None
        self.last_input_tokens: int = 0

    # ---- question builders ---------------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no question. The answer is P(true)."""
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = dict(criteria)
        return q

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N options. `criteria` maps option name to what it means."""
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking --------------------------------------------------------------------

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"state": state, "questions": dict(questions)}).encode()
        req = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise DeciderUnavailable(f"no decider at {self.url}: {exc}") from exc
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_server_latency_ms = payload.get("latency_ms")
        self.last_input_tokens = payload.get("usage", {}).get("input_tokens", 0)
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise DeciderUnavailable(f"no decider at {self.url}: {exc}") from exc
