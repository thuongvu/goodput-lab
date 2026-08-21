#!/usr/bin/env python3
"""Compare timed_trace jsonl request count to vLLM bench input_lens / output_lens."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CountMatch:
    """Counts from a timed_trace jsonl vs a bench result json.

    timed_trace_count: non-empty lines in the jsonl.
    input_count: len of input_lens on the bench object.
    output_count: len of output_lens on the bench object.
    Missing or empty output_lens is output_count=0 and is not a mismatch.
    """

    timed_trace_count: int
    input_count: int
    output_count: int

    @property
    def ok(self) -> bool:
        if self.input_count != self.timed_trace_count:
            return False
        if self.output_count and self.output_count != self.timed_trace_count:
            return False
        return True


def from_paths(timed_trace: Path, bench: Path) -> CountMatch:
    """Read timed_trace jsonl and bench json; return CountMatch counts.

    timed_trace: local jsonl path; count is non-empty lines. bench: whole-file
    JSON, or last non-empty line if that decode fails (JSONL).
    """
    timed_trace_count = count_nonempty_lines(timed_trace)
    payload = _load_bench(bench)
    input_count = len(payload.get("input_lens") or [])
    output_count = len(payload.get("output_lens") or [])
    return CountMatch(
        timed_trace_count=timed_trace_count,
        input_count=input_count,
        output_count=output_count,
    )


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines in path (timed_trace jsonl)."""
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _load_bench(path: Path) -> dict:
    """Parse bench JSON. Whole file, or last non-empty line if that is JSONL."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last = None
        for line in text.splitlines():
            line = line.strip()
            if line:
                last = json.loads(line)
        if last is None:
            raise ValueError("empty bench {}".format(path))
        return last


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: timed_trace jsonl path, then bench result json path.

    --count prints the timed_trace nonempty line count and skips the bench.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        action="store_true",
        help="print nonempty line count of timed_trace jsonl",
    )
    parser.add_argument("timed_trace", type=Path)
    parser.add_argument("bench", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.count:
        if args.bench is not None:
            parser.error("--count does not take a bench path")
    elif args.bench is None:
        parser.error("bench path required (or pass --count)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Print count, or count-match OK / HARD STOP; nonzero on mismatch or empty bench."""
    args = parse_args(argv)
    if args.count:
        print(count_nonempty_lines(args.timed_trace))
        return 0
    try:
        match = from_paths(args.timed_trace, args.bench)
    except ValueError as err:
        print("HARD STOP: {}".format(err), file=sys.stderr)
        return 1
    if not match.ok:
        print(
            "HARD STOP: bench input_count={} output_count={} timed_trace_count={}".format(
                match.input_count, match.output_count, match.timed_trace_count
            ),
            file=sys.stderr,
        )
        return 1
    print("count match OK timed_trace_count={}".format(match.timed_trace_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
