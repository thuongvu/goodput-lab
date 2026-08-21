#!/usr/bin/env python3
"""Write a timed-trace jsonl for the decode load.

Decode load is short prompt, long generate: stay in decode.
Each row is timestamp, input_length, output_length, hash_ids, and phase.

Phase is an intention tag for a later `python3 -m goodput_lab.results` join.
Rows use the decode token-length envelope. Each phase spaces sends by its
interarrival: seconds between consecutive sends.
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from goodput_lab.timed_trace import CHUNK_HASH_SIZE, dummy_chunk_ids


# PROFILES
# Named workloads. decode-envelope is implemented.


class Profile(enum.Enum):
    """Named workload profiles."""

    DECODE_ENVELOPE = "decode-envelope"
    PREFILL_BLAST = "prefill-blast"
    OWNERSHIP = "ownership"


# PHASES
# Intention tags on generated jsonl rows for the results join.
# Interarrival is seconds between consecutive sends, constant within a phase.


class Phase(enum.Enum):
    """Intention tag on a generated jsonl row for the results join.

    healthy/busy/pressure/recovery. The tag is the intended load and the GPU may
    behave differently.
    """

    HEALTHY = "healthy"
    BUSY = "busy"
    PRESSURE = "pressure"
    RECOVERY = "recovery"


# DECODE SHAPE
# Prompt and output token ranges, plus per-phase interarrival (seconds).
# pin.yaml stays engine identity; retune interarrival here if the first GPU call misses.


@dataclass(frozen=True)
class DecodeShape:
    """Token ranges and per-phase interarrivals (seconds between sends) for the decode load.

    Short prompt, long generate. Each phase spaces sends by that phase's
    interarrival, in seconds between consecutive sends.
    """

    input_len_min: int
    input_len_max: int
    output_len_min: int
    output_len_max: int
    phase_duration: float
    interarrival: dict[Phase, float]


DECODE_SHAPE = DecodeShape(
    input_len_min=96,
    input_len_max=192,
    output_len_min=256,
    output_len_max=384,
    phase_duration=30.0,
    interarrival={
        Phase.HEALTHY: 2.0,
        Phase.BUSY: 0.5,
        Phase.PRESSURE: 0.15,
        Phase.RECOVERY: 2.0,
    },
)


# GENERATE ROWS
# decode_profile_rows builds in-memory jsonl dicts. Four vLLM keys plus phase.


TIMED_TRACE_KEYS = (
    "timestamp",
    "input_length",
    "output_length",
    "hash_ids",
)


def decode_profile_rows(
    seed: int, shape: DecodeShape = DECODE_SHAPE
) -> list[dict]:
    """Build decode jsonl row dicts in memory. Main writes the file."""
    rng = random.Random(seed)
    rows: list[dict] = []
    next_hash = 1
    phase_start = 0.0
    for phase in Phase:
        gap = shape.interarrival[phase]
        count = int(math.floor(shape.phase_duration / gap + 1e-9))
        for step in range(count):
            input_length = rng.randint(shape.input_len_min, shape.input_len_max)
            output_length = rng.randint(
                shape.output_len_min, shape.output_len_max
            )
            hash_ids = dummy_chunk_ids(input_length, CHUNK_HASH_SIZE, next_hash)
            next_hash += len(hash_ids)
            rows.append(
                {
                    "timestamp": phase_start + step * gap,
                    "input_length": input_length,
                    "output_length": output_length,
                    "hash_ids": hash_ids,
                    "phase": phase.value,
                }
            )
        phase_start += shape.phase_duration
    return rows


# WRITE JSONL
# write_jsonl dumps one JSON object per line.


def write_jsonl(path: Path, rows: list[dict]) -> int:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


# CLI
# --profile, --seed, --out. decode-envelope writes jsonl.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: --profile, --seed, --out. decode-envelope is the implemented decode load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="decode-envelope | prefill-blast | ownership",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse args and write jsonl for a profile."""
    args = parse_args(argv)
    try:
        profile = Profile(args.profile)
    except ValueError:
        print("unknown profile: {}".format(args.profile), file=sys.stderr)
        return 2
    if profile != Profile.DECODE_ENVELOPE:
        print("not yet: {}".format(profile.value), file=sys.stderr)
        return 1

    rows = decode_profile_rows(args.seed)
    n_written = write_jsonl(args.out, rows)
    print(
        "wrote {} requests -> {}\nprofile={} seed={}".format(
            n_written,
            args.out,
            profile.value,
            args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
