#!/usr/bin/env python3
"""jsonl nonempty line count vs vLLM bench.json input_lens, so the later index join is 1:1."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CountMatch:
    """Counts from a timed-trace jsonl vs the vLLM ServeBenchmark dump (bench.json).

    timed_trace_count is nonempty jsonl lines. input_count is len(input_lens).
    output_count is len(output_lens). Missing or empty output_lens is 0 and still matches.
    """

    timed_trace_count: int
    input_count: int
    output_count: int

    @property
    def ok(self) -> bool:
        """True when input_lens (and nonempty output_lens) match the jsonl line count."""
        if self.input_count != self.timed_trace_count:
            return False
        if self.output_count and self.output_count != self.timed_trace_count:
            return False
        return True


def from_paths(timed_trace: Path, bench: Path) -> CountMatch:
    """Read jsonl and bench.json; return line counts vs input_lens / output_lens."""
    timed_trace_count = count_nonempty_lines(timed_trace)
    payload = json.loads(bench.read_text())
    input_count = len(payload.get("input_lens") or [])
    output_count = len(payload.get("output_lens") or [])
    return CountMatch(
        timed_trace_count=timed_trace_count,
        input_count=input_count,
        output_count=output_count,
    )


def count_nonempty_lines(path: Path) -> int:
    """Count nonempty lines in the served jsonl. That count must match bench input_lens."""
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: jsonl path, then bench.json. --count prints the jsonl line count and skips the bench."""
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
    """Print count, or count-match OK. run.sh after bench; nonzero on mismatch."""
    args = parse_args(argv)
    if args.count:
        print(count_nonempty_lines(args.timed_trace))
        return 0
    try:
        match = from_paths(args.timed_trace, args.bench)
    except ValueError as err:
        print("count_match: {}".format(err), file=sys.stderr)
        return 1
    if not match.ok:
        print(
            "count_match: jsonl rows {} != bench input_lens {}".format(
                match.timed_trace_count, match.input_count
            ),
            file=sys.stderr,
        )
        return 1
    print("count match OK timed_trace_count={}".format(match.timed_trace_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
