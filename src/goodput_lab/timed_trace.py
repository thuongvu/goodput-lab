#!/usr/bin/env python3
"""Write a vLLM timed_trace JSONL from a RequestWindow.

The write path: drop prompts longer than the model max; optionally keep only
the long-prompt tail; keep every stride-th request (1 = keep all); convert
to TimedTraceRecord; write jsonl.

Optional second file: keep only the long-prompt tail and set output_length=1.
That is the HOL canary, not the main mix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from goodput_lab.request_window import (
    RecordedRequest,
    DEFAULT_CSV,
    check_window_args,
    download_azure_csv,
    fits_max_model_len,
    from_csv,
    resolve_window_bounds,
)
from goodput_lab.window_policy import load_profile

CHUNK_HASH_SIZE = 16  # tokens per dummy hash_id; must match run.sh


@dataclass(frozen=True)
class TimedTraceRecord:
    """One vLLM timed_trace JSONL request.

    timestamp is seconds from the first request x arrival_scale, not the Azure
    invocation datetime. Stock vLLM field name; must stay.
    https://docs.vllm.ai/en/stable/cli/bench/serve/
    """

    timestamp: float  # relative seconds after arrival_scale; not Azure invocation datetime
    input_length: int
    output_length: int
    hash_ids: list[int]
    azure_row: int
    azure_generated: int


def build_timed_trace(
    rows: list[RecordedRequest],
    *,
    max_model_len: int,
    arrival_scale: float,
    stride: int,
    keep_long_prompt_tail: bool = False,
    chunk_size: int = CHUNK_HASH_SIZE,
) -> list[TimedTraceRecord]:
    """Main entry: drop -> optional long-prompt tail -> downsample -> convert. Caller writes jsonl.

    Drops prompts that cannot fit max_model_len, optionally keeps the
    long-prompt tail (output_length=1), keeps every stride-th request
    (1 = keep all), converts to TimedTraceRecord. No write.
    """
    after_fit, _counts = drop_over_max_model_len(rows, max_model_len)
    if not after_fit:
        raise ValueError("no rows after max_model_len drop")

    output_len = None
    rows = after_fit
    if keep_long_prompt_tail:
        rows = _keep_long_prompt_tail(rows, HOL_CONTEXT_PCTL)
        output_len = 1
        if not rows:
            raise ValueError("no rows after keep_long_prompt_tail filter")

    rows = downsample(rows, stride)
    if not rows:
        raise ValueError("no rows after downsample")

    return to_timed_trace_records(
        rows,
        arrival_scale=arrival_scale,
        chunk_size=chunk_size,
        output_len=output_len,
    )


def drop_over_max_model_len(
    rows: list[RecordedRequest], max_model_len: int
) -> tuple[list[RecordedRequest], dict]:
    """Drop empty prompts and rows with input_length + output_length > max_model_len."""
    after_fit: list[RecordedRequest] = []
    n_dropped_empty = 0
    n_dropped_over_max_model_len = 0
    for row in rows:
        if not fits_max_model_len(row, max_model_len):
            if row.input_length <= 0:
                n_dropped_empty += 1
            else:
                n_dropped_over_max_model_len += 1
            continue
        after_fit.append(row)
    counts = {
        "n_dropped_empty": n_dropped_empty,
        "n_dropped_over_max_model_len": n_dropped_over_max_model_len,
        "n_after_fit": len(after_fit),
    }
    return after_fit, counts


HOL_CONTEXT_PCTL = 90


def _keep_long_prompt_tail(
    rows: list[RecordedRequest], p: float = HOL_CONTEXT_PCTL
) -> list[RecordedRequest]:
    """Keep the long-prompt tail as a cheap HOL canary.

    Long prompts can block shorter ones at the head of the queue. This keeps
    rows with input_length >= the p-th percentile of input_length.
    output_length is forced to 1 later. Not a quality filter on the main mix.
    """
    if not rows:
        return []
    thresh = _percentile(sorted(r.input_length for r in rows), p)
    return [r for r in rows if r.input_length >= thresh]


def _percentile(sorted_vals: list[int], p: float) -> float:
    """Long-prompt cutoff for _keep_long_prompt_tail.

    Input is already sorted. Linear interpolation between neighbors.
    Empty list -> NaN.
    """
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def downsample(rows: list[RecordedRequest], stride: int) -> list[RecordedRequest]:
    """Keep every stride-th request. stride=1 keeps all. Standard downsample stride, not a count."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return list(rows[::stride])


def to_timed_trace_records(
    rows: list[RecordedRequest],
    *,
    arrival_scale: float,
    chunk_size: int,
    output_len: int | None = None,
) -> list[TimedTraceRecord]:
    """Convert RequestWindow rows to TimedTraceRecord list only.

    Relative timestamps x arrival_scale, plus hash_ids. No drop, no downsample,
    no write.
    """
    if not rows:
        return []
    t0 = rows[0].arrival_time
    next_hash = 1
    records: list[TimedTraceRecord] = []
    for row in rows:
        hash_ids = dummy_chunk_ids(row.input_length, chunk_size, next_hash)
        next_hash += len(hash_ids)
        records.append(
            TimedTraceRecord(
                timestamp=(row.arrival_time - t0).total_seconds() * arrival_scale,
                input_length=row.input_length,
                output_length=output_len if output_len is not None else row.output_length,
                hash_ids=hash_ids,
                azure_row=row.index,
                azure_generated=row.output_length,
            )
        )
    return records


def dummy_chunk_ids(input_length: int, chunk_size: int, start_hash: int) -> list[int]:
    """Dummy-prompt hash_ids for vLLM timed_trace. TimedTrace expands these to input_length tokens."""
    if input_length <= 0:
        return []
    n = (input_length + chunk_size - 1) // chunk_size
    return list(range(start_hash, start_hash + n))


def write_jsonl(path: Path, records: list[TimedTraceRecord]) -> int:
    """Write one JSON object per line. Returns n written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    return len(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags. Pipeline helpers take data, not argparse objects.

    --profile supplies csv, window, apply knobs, and max_model_len. Explicit
    flags override those fields; the profile fills the rest. Prefer the profile
    unless a flag is passed.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=(
            "Replay profile JSON from window_policy.py. Supplies csv, start/end, "
            "apply.arrival_scale, apply.stride, and max_model_len. "
            "An explicit flag wins over the profile for that field."
        ),
    )
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--window-sec", type=float, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "vLLM max_model_len (GPU fit). Drop requests with "
            "input_length + output_length above this bound. "
            "Required unless --profile already has it."
        ),
    )
    p.add_argument(
        "--arrival-scale",
        type=float,
        default=None,
        help=(
            "Scale inter-arrivals: multiply JSONL timestamps. "
            "Larger = slower arrivals = less load. "
            "Not --timed-trace-sec-multiplier. "
            "Overrides profile apply.arrival_scale when passed."
        ),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        dest="stride",
        help=(
            "Keep every stride-th request after the max_model_len drop. "
            "Standard downsample stride (1 = keep all), not a count of rows to keep. "
            "Overrides profile apply.stride when passed. "
            "The tool usually chooses this; pass only to override the profile."
        ),
    )
    p.add_argument(
        "--chunk-hash-size",
        type=int,
        default=CHUNK_HASH_SIZE,
        help=(
            "Tokens per hash_id. vLLM timed_trace default 16 (Qwen/Alibaba); "
            "Mooncake traces use 512."
        ),
    )
    p.add_argument(
        "--keep-long-prompt-tail",
        action="store_true",
        help="HOL canary: separate long-prompt tail file (1 output token). Not the main mix.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta-out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.profile is None and args.max_model_len is None:
        p.error("--max-model-len is required unless --profile is set")
    try:
        check_window_args(args.start, args.end, args.window_sec)
    except ValueError as err:
        p.error(str(err))
    return args


def resolve_replay(args: argparse.Namespace) -> argparse.Namespace:
    """Fill csv/window/apply from --profile. Explicit flags already on args win."""
    if args.profile is not None:
        _fill_from_profile(args, load_profile(args.profile))
    if args.csv is None:
        args.csv = DEFAULT_CSV
    if args.arrival_scale is None:
        args.arrival_scale = 1.0
    if args.stride is None:
        args.stride = 1
    return args


def _fill_from_profile(args: argparse.Namespace, profile: dict) -> None:
    """Copy missing fields from a replay profile. Do not overwrite explicit flags."""
    if args.csv is None and profile.get("csv"):
        args.csv = Path(profile["csv"])
    if args.start is None and profile.get("start"):
        args.start = profile["start"]
    if args.end is None and profile.get("end"):
        args.end = profile["end"]
    apply = profile.get("apply") or {}
    if args.arrival_scale is None and "arrival_scale" in apply:
        args.arrival_scale = float(apply["arrival_scale"])
    if args.stride is None and "stride" in apply:
        args.stride = int(apply["stride"])
    if args.max_model_len is None:
        max_model_len = apply.get("max_model_len")
        if max_model_len is None:
            max_model_len = (profile.get("inputs") or {}).get("max_model_len")
        if max_model_len is not None:
            args.max_model_len = int(max_model_len)


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, write jsonl."""
    args = resolve_replay(parse_args(argv))
    if args.max_model_len is None:
        print(
            "--max-model-len is required (or set it in the profile)",
            file=sys.stderr,
        )
        return 2
    if args.arrival_scale <= 0:
        print("--arrival-scale must be > 0", file=sys.stderr)
        return 2
    if args.stride < 1:
        print("--stride must be >= 1", file=sys.stderr)
        return 2
    if args.chunk_hash_size < 1:
        print("--chunk-hash-size must be >= 1", file=sys.stderr)
        return 2

    csv_path = download_azure_csv(args.csv) if args.download else args.csv
    if not csv_path.exists():
        print("missing {}".format(csv_path), file=sys.stderr)
        return 2

    try:
        start, end = resolve_window_bounds(
            csv_path, args.start, args.end, args.window_sec
        )
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    window = from_csv(csv_path, start, end, args.max_rows)

    if args.keep_long_prompt_tail:
        print(
            "HOL canary: separate file, not the main mix",
            file=sys.stderr,
        )

    try:
        records = build_timed_trace(
            window.rows,
            max_model_len=args.max_model_len,
            arrival_scale=args.arrival_scale,
            stride=args.stride,
            keep_long_prompt_tail=args.keep_long_prompt_tail,
            chunk_size=args.chunk_hash_size,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    n_written = write_jsonl(args.out, records)

    by_index = {r.index: r for r in window.rows}
    t0 = by_index[records[0].azure_row].arrival_time
    t_last = by_index[records[-1].azure_row].arrival_time
    t_raw = (t_last - t0).total_seconds()
    runtime = t_raw * args.arrival_scale
    meta = {
        "csv": str(csv_path),
        "out": str(args.out),
        "max_model_len": args.max_model_len,
        "n_written": n_written,
        "arrival_scale": args.arrival_scale,
        "stride": args.stride,
        "chunk_hash_size": args.chunk_hash_size,
        "sec_multiplier_must_be": 1,
        "keep_long_prompt_tail": args.keep_long_prompt_tail,
        "t0": t0.isoformat(),
        "raw_span_seconds": t_raw,
        "runtime": runtime,
    }
    meta_path = args.meta_out or args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        (
            "wrote {} requests -> {}\n"
            "arrival_scale={} stride={} "
            "chunk={}\n"
            "raw_span={:.3f}s runtime={:.3f}s\n"
            "sec_multiplier must be 1\n"
            "meta={}"
        ).format(
            n_written,
            args.out,
            args.arrival_scale,
            args.stride,
            args.chunk_hash_size,
            t_raw,
            runtime,
            meta_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
