#!/usr/bin/env python3
"""Load the vLLM ServeBenchmark dump (bench.json).

One request is one index. The lab join (results.py) maps ttfts[i] to jsonl
row i and writes results.json. count_match.py checks jsonl nonempty line
count vs input_lens length before that join. start_times on this vLLM image
are time.perf_counter() seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# vLLM ServeBenchmark dump (bench.json). The lab join (results.py) maps ttfts[i] to jsonl row i.


@dataclass(frozen=True)
class BenchResult:
    """Per-request arrays from the vLLM ServeBenchmark dump (bench.json)."""

    input_lens: list
    output_lens: list
    ttfts: list
    itls: list
    queue_times: list | None
    start_times: list | None = None
    duration: float | None = None


def from_json(path: Path) -> BenchResult:
    """Parse per-request arrays from a vLLM bench.json object."""
    payload = json.loads(path.read_text())
    return BenchResult(
        input_lens=_required_list(payload, "input_lens"),
        output_lens=list(payload.get("output_lens") or []),
        ttfts=_required_list(payload, "ttfts"),
        itls=_required_list(payload, "itls"),
        queue_times=payload.get("queue_times"),
        start_times=payload.get("start_times"),
        duration=payload.get("duration"),
    )


def _required_list(payload: dict, key: str) -> list:
    """Return payload[key] as a list."""
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(
            "{} must be a list, got {}".format(key, type(value).__name__)
        )
    return value
