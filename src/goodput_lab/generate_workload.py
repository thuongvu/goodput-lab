#!/usr/bin/env python3
"""Write a timed-trace jsonl for a named workload profile.

Each row is timestamp, input_length, output_length, hash_ids, and phase.
Phase is an intention tag for a later `python3 -m goodput_lab.results` join.
decode-envelope spaces sends by per-phase interarrival on the decode
token-length envelope. prefill-blast schedules decode_stream, long_prefill,
and arrival_probe rows. ownership writes those same rows; owners are serve
settings passed on run.sh --diff.
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
# Named workloads.


class Profile(enum.Enum):
    """Named workload profiles."""

    DECODE_ENVELOPE = "decode-envelope"
    PREFILL_BLAST = "prefill-blast"
    OWNERSHIP = "ownership"


# PHASES
# Intention tags on generated jsonl rows for the results join.
# decode-envelope: healthy/busy/pressure/recovery, constant interarrival
# within a phase. prefill-blast and ownership: decode_stream/long_prefill/
# arrival_probe on the same key.


class Phase(enum.Enum):
    """Intention tag on a generated jsonl row for the results join.

    healthy/busy/pressure/recovery. The tag is the intended load and the GPU may
    behave differently.
    """

    HEALTHY = "healthy"
    BUSY = "busy"
    PRESSURE = "pressure"
    RECOVERY = "recovery"


class PrefillRole(enum.Enum):
    """Prefill-blast intention tag written on the jsonl phase key.

    decode_stream is in-flight decode (short prompt, long generate).
    long_prefill is a fat prompt with output length 1. arrival_probe is a
    tiny new arrival, a time-to-first-token canary.
    """

    DECODE_STREAM = "decode_stream"
    LONG_PREFILL = "long_prefill"
    ARRIVAL_PROBE = "arrival_probe"


# DECODE SHAPE
# Prompt and output token ranges, plus per-phase interarrival (seconds).
# pin.yaml stays engine identity.


@dataclass(frozen=True)
class DecodeShape:
    """Token ranges and per-phase interarrivals (seconds between sends) for the decode load.

    Short prompt, long generate so the run stays in decode. Lengths are the same
    every phase; each phase spaces sends by that phase's interarrival.
    """

    input_len_min: int
    input_len_max: int
    output_len_min: int
    output_len_max: int
    phase_duration: float
    interarrival: dict[Phase, float]


# Healthy 2.0 s is low concurrency so time to first token and inter-token
# latency are a quiet baseline. Busy 0.5 s raises in-flight load and is
# still meant to stay healthy. Pressure 0.08 s is tight enough that many
# requests share decode. Recovery 2.0 s matches healthy so the tail can drain.
DECODE_SHAPE = DecodeShape(
    input_len_min=96,
    input_len_max=192,
    output_len_min=256,
    output_len_max=384,
    phase_duration=30.0,
    interarrival={
        Phase.HEALTHY: 2.0,
        Phase.BUSY: 0.5,
        Phase.PRESSURE: 0.08,
        Phase.RECOVERY: 2.0,
    },
)


# PREFILL SHAPE
# Token bands and send schedule. Probes pack onto the packed prefills.


@dataclass(frozen=True)
class PrefillShape:
    """Token bands and send schedule for in-flight decode, long prefills, and tiny arrival probes.

    Decode streams keep generating while large prefills (output length 1) and
    tiny arrival probes land on the same timeline.
    """

    decode_stream_input_len_min: int
    decode_stream_input_len_max: int
    decode_stream_output_len_min: int
    decode_stream_output_len_max: int
    long_prefill_input_len_min: int
    long_prefill_input_len_max: int
    long_prefill_output_len: int
    arrival_probe_input_len_min: int
    arrival_probe_input_len_max: int
    arrival_probe_output_len_min: int
    arrival_probe_output_len_max: int
    decode_stream_count: int
    decode_stream_interarrival: float
    long_prefill_count: int
    long_prefill_delay: float
    long_prefill_interarrival: float
    arrival_probe_count: int
    first_arrival_probe: float
    arrival_probe_interarrival: float


# Decode-stream generate is long enough that streams are still producing
# tokens when the prefills land. Prefills are a large prompt with output
# length 1 that fits 32k context. Arrival probes are tiny and packed onto
# that prefill window at a 0.15 s send gap so new arrivals see the prefill.
PREFILL_SHAPE = PrefillShape(
    decode_stream_input_len_min=96,
    decode_stream_input_len_max=192,
    decode_stream_output_len_min=1024,
    decode_stream_output_len_max=1280,
    long_prefill_input_len_min=8192,
    long_prefill_input_len_max=12288,
    long_prefill_output_len=1,
    arrival_probe_input_len_min=16,
    arrival_probe_input_len_max=32,
    arrival_probe_output_len_min=8,
    arrival_probe_output_len_max=16,
    decode_stream_count=12,
    decode_stream_interarrival=0.25,
    long_prefill_count=3,
    long_prefill_delay=0.5,
    long_prefill_interarrival=2.0,
    arrival_probe_count=32,
    first_arrival_probe=3.25,
    arrival_probe_interarrival=0.15,
)


# OWNERSHIP SERVE
# Same jsonl as prefill-blast. Owners are sequence cap, KV capacity, and
# preemption. pin.yaml does not carry these knobs; run.sh --diff appends
# them to vllm serve.


@dataclass(frozen=True)
class OwnershipServe:
    """Intended vLLM serve flags for one owner. Human passes them on run.sh --diff."""

    name: str
    max_num_seqs: int | None
    gpu_memory_utilization: float | None

    def diff_argv(self) -> list[str]:
        """Extra vllm serve tokens for run.sh --diff. Empty means stock serve."""
        argv: list[str] = []
        if self.max_num_seqs is not None:
            argv.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.gpu_memory_utilization is not None:
            argv.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.gpu_memory_utilization),
                ]
            )
        return argv


# sequence_cap: 8 is below the 12 in-flight decode streams so running can
# glue at the cap. kv_capacity: stock serve. preemption: 0.45 is a starting
# KV shrink; calibrate to the mildest gpu_memory_utilization that produces
# a nonzero preemption delta.
SEQUENCE_CAP_SERVE = OwnershipServe(
    name="sequence_cap",
    max_num_seqs=8,
    gpu_memory_utilization=None,
)
KV_CAPACITY_SERVE = OwnershipServe(
    name="kv_capacity",
    max_num_seqs=None,
    gpu_memory_utilization=None,
)
PREEMPTION_SERVE = OwnershipServe(
    name="preemption",
    max_num_seqs=None,
    gpu_memory_utilization=0.45,
)
OWNERSHIP_SERVE = (
    SEQUENCE_CAP_SERVE,
    KV_CAPACITY_SERVE,
    PREEMPTION_SERVE,
)


# GENERATE ROWS
# Generate in-memory jsonl dicts.


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


def prefill_blast_rows(
    seed: int, shape: PrefillShape = PREFILL_SHAPE
) -> list[dict]:
    """Build prefill-blast jsonl row dicts in memory. Main writes the file.

    phase is decode_stream, long_prefill, or arrival_probe.
    """
    rng = random.Random(seed)
    specs: list[tuple[float, PrefillRole, int, int]] = []
    for step in range(shape.decode_stream_count):
        specs.append(
            (
                step * shape.decode_stream_interarrival,
                PrefillRole.DECODE_STREAM,
                rng.randint(
                    shape.decode_stream_input_len_min,
                    shape.decode_stream_input_len_max,
                ),
                rng.randint(
                    shape.decode_stream_output_len_min,
                    shape.decode_stream_output_len_max,
                ),
            )
        )
    last_decode_stream = (
        shape.decode_stream_count - 1
    ) * shape.decode_stream_interarrival
    for step in range(shape.long_prefill_count):
        specs.append(
            (
                last_decode_stream
                + shape.long_prefill_delay
                + step * shape.long_prefill_interarrival,
                PrefillRole.LONG_PREFILL,
                rng.randint(
                    shape.long_prefill_input_len_min,
                    shape.long_prefill_input_len_max,
                ),
                shape.long_prefill_output_len,
            )
        )
    for step in range(shape.arrival_probe_count):
        specs.append(
            (
                shape.first_arrival_probe
                + step * shape.arrival_probe_interarrival,
                PrefillRole.ARRIVAL_PROBE,
                rng.randint(
                    shape.arrival_probe_input_len_min,
                    shape.arrival_probe_input_len_max,
                ),
                rng.randint(
                    shape.arrival_probe_output_len_min,
                    shape.arrival_probe_output_len_max,
                ),
            )
        )
    specs.sort(key=lambda spec: (spec[0], spec[1].value))
    rows: list[dict] = []
    next_hash = 1
    for timestamp, phase, input_length, output_length in specs:
        hash_ids = dummy_chunk_ids(input_length, CHUNK_HASH_SIZE, next_hash)
        next_hash += len(hash_ids)
        rows.append(
            {
                "timestamp": timestamp,
                "input_length": input_length,
                "output_length": output_length,
                "hash_ids": hash_ids,
                "phase": phase.value,
            }
        )
    return rows


def ownership_rows(seed: int, shape: PrefillShape = PREFILL_SHAPE) -> list[dict]:
    """Same rows as prefill-blast. Owners are serve settings."""
    return prefill_blast_rows(seed, shape)


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
# --profile, --seed, --out. ownership prints intended run.sh --diff lines.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: --profile, --seed, --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="decode-envelope | prefill-blast | ownership",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _print_ownership_serve() -> None:
    """Print that rows match prefill-blast and the three intended run.sh --diff lines."""
    print("same rows as prefill-blast; owners are serve settings")
    for variant in OWNERSHIP_SERVE:
        argv = variant.diff_argv()
        if argv:
            print("{}: --diff {}".format(variant.name, " ".join(argv)))
        else:
            print("{}: stock serve (no --diff)".format(variant.name))


def main(argv: list[str] | None = None) -> int:
    """Parse args and write jsonl for a profile."""
    args = parse_args(argv)
    try:
        profile = Profile(args.profile)
    except ValueError:
        print("unknown profile: {}".format(args.profile), file=sys.stderr)
        return 2
    if profile == Profile.DECODE_ENVELOPE:
        rows = decode_profile_rows(args.seed)
    elif profile == Profile.PREFILL_BLAST:
        rows = prefill_blast_rows(args.seed)
    elif profile == Profile.OWNERSHIP:
        rows = ownership_rows(args.seed)
    else:
        print("not yet: {}".format(profile.value), file=sys.stderr)
        return 1

    n_written = write_jsonl(args.out, rows)
    print(
        "wrote {} requests -> {}\nprofile={} seed={}".format(
            n_written,
            args.out,
            profile.value,
            args.seed,
        )
    )
    if profile == Profile.OWNERSHIP:
        _print_ownership_serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
