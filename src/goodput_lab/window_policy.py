#!/usr/bin/env python3
"""Recommend arrival_scale / stride / estimated_runtime given max_runtime.

This file recommends arrival_scale, stride, and estimated_runtime given
max_runtime. It is read-only on the RequestWindow. Writes a JSON replay
profile; timed_trace.py --profile applies the apply knobs when writing JSONL.

Unscaled arrival rate is requests/sec as logged, not after --arrival-scale.
Larger arrival_scale = lower offered load.

You pass what you know:
  --csv                               which log (optional; has a default)
  --window-sec N                      first N seconds of the CSV (smoke path)
  --start / --end                     a specific slice once you know it
  --max-runtime                       GPU seconds this replay may take
                                      (default 1200 = 20 min smoke)
  --out                               profile path
  optional --max-model-len, --download, --min-sample-size
  optional --guessed-rate             after a first GPU run, not a QPS you
                                      invent; omit for time-fit only

Omit --guessed-rate: time-fit only. Choose arrival_scale / stride so
estimated_runtime fits max_runtime; do not target an offered QPS.
If window_span <= max_runtime: arrival_scale can be 1 (stride=1).
If the window is longer than the budget: stride (and/or scale) so it fits.

If --guessed-rate is set: scale toward that offered rate, then stride to
fit time. Bootstrap from one GPU data point, not from thin air.

The tool writes apply knobs (arrival_scale, stride) into the profile.
build_profile / main read argparse into PolicyInputs; the human does not
construct WindowPolicy.

Azure Conversation trace: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
vLLM --max-model-len (prompt + output bound): optional here; required in timed_trace.py
unless the profile already has it.
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from goodput_lab.request_window import (
    DEFAULT_CSV,
    RecordedRequest,
    RequestWindow,
    check_window_args,
    download_azure_csv,
    fits_max_model_len,
    from_csv,
    resolve_window_bounds,
)


@dataclass(frozen=True)
class PolicyInputs:
    """Measured window plus CLI budget knobs for recommend_policy.

    window_span: seconds from first to last arrival_time in the log window.
    sample_size: request count after optional max_model_len drop, before stride.
    guessed_rate: optional target offered req/s (after a first GPU run).
                 None = time-fit only; do not target an offered QPS.
    max_runtime: GPU time budget.
    min_sample_size: floor after stride so the mix isn't tiny.
    """

    window_span: float
    sample_size: int
    guessed_rate: float | None
    max_runtime: float
    min_sample_size: int

    @property
    def unscaled_rate(self) -> float:
        """sample_size / window_span (req/s as logged)."""
        if self.window_span <= 0:
            return float("inf")
        return self.sample_size / self.window_span


class PolicyReason(str, enum.Enum):
    KEEP_ALL_FITS = "keep_all_fits"
    SCALE_ONLY = "scale_only"
    DOWNSAMPLE = "downsample"
    NEED_LONGER_RAW_WINDOW = "need_longer_raw_window"
    SHRINK_MIN_SAMPLE_SIZE = "shrink_min_sample_size"


# Policy: recommend arrival_scale, stride, estimated_runtime given max_runtime.
@dataclass(frozen=True)
class WindowPolicy:
    """Decision output for one window. timed_trace.py applies it.

    Fields: reason, arrival_scale, stride, estimated_runtime, plus
    after-downsample arrival_scale / estimated_runtime / sample_size when
    stride > 1. Do not copy PolicyInputs (guessed_rate, max_runtime,
    min_sample_size, window_span, sample_size) onto this record.

    reason: PolicyReason.
    stride: keep every stride-th request (1 = keep all). Standard
    downsample stride, not a count of rows to keep.
    arrival_scale: larger = lower offered load.
    Profile apply.arrival_scale is arrival_scale_after_downsample if
    stride > 1 else arrival_scale; apply.stride is stride.
    timed_trace must use apply (do not also scale by stride).
    """

    reason: PolicyReason
    arrival_scale: float
    stride: int
    estimated_runtime: float
    arrival_scale_after_downsample: float | None = None
    estimated_runtime_after_downsample: float | None = None
    sample_size_after_downsample: int | None = None

    def to_dict(self) -> dict:
        """JSON-ready dict. Omits unused after-downsample fields."""
        data = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        data["reason"] = self.reason.value
        return data


def recommend_policy(inputs: PolicyInputs) -> WindowPolicy:
    """Recommend arrival_scale, stride, and estimated_runtime that fit max_runtime.

    stride: keep every stride-th request (1 = keep all). Standard downsample
    stride, not a count of rows to keep. Shortens GPU replay.
    estimated_runtime = estimated GPU wall time of the replay after arrival_scale
    (and after stride if stride > 1).
    max_runtime = GPU time budget: do not recommend a replay longer than this.
    arrival_scale: larger = lower offered load.

    unscaled_rate is sample_size / window_span (req/s as logged). guessed_rate is an
    optional target offered rate (after a first GPU run). When set, arrival_scale is
    roughly unscaled_rate / guessed_rate (larger scale = slower replay = lower load).
    When omitted (None): time-fit only. arrival_scale starts at 1; do not target QPS.

    Caller measures window_span and sample_size on the RequestWindow, then this function
    chooses arrival_scale / stride / estimated_runtime. Those measurements are inputs,
    not CLI knobs like max_runtime / min_sample_size.
    """
    if inputs.guessed_rate is not None and inputs.guessed_rate <= 0:
        raise ValueError("guessed_rate must be > 0")

    if inputs.guessed_rate is None:
        arrival_scale = 1.0
    else:
        unscaled_rate = inputs.unscaled_rate
        arrival_scale = (
            unscaled_rate / inputs.guessed_rate if unscaled_rate > 0 else float("inf")
        )
    estimated_runtime = inputs.window_span * arrival_scale

    # Too few requests even before stride: widen the raw window.
    if inputs.sample_size < inputs.min_sample_size:
        return _need_longer_window(inputs, arrival_scale)

    # Estimated GPU replay already fits: keep every request (stride=1).
    if estimated_runtime <= inputs.max_runtime:
        return _fit_without_downsample(inputs, arrival_scale)

    # Estimated GPU replay time (window_span x arrival_scale) exceeds max_runtime, so pick
    # the smallest stride (keep every stride-th) that fits the budget and
    # min_sample_size.
    stride, sample_size_after_downsample, estimated_runtime_after_downsample = (
        _choose_stride(inputs, estimated_runtime)
    )
    if (
        estimated_runtime_after_downsample <= inputs.max_runtime
        and sample_size_after_downsample >= inputs.min_sample_size
    ):
        return _downsample_policy(inputs, arrival_scale, stride)

    # Cannot meet both max_runtime and min_sample_size: last resort, shrink min_sample_size.
    return _shrink_min_sample_size(inputs, arrival_scale, stride)


def _fill_policy(
    inputs: PolicyInputs,
    *,
    reason: PolicyReason,
    arrival_scale: float,
    stride: int,
) -> WindowPolicy:
    """Fill estimated_runtime and after-stride fields from inputs+stride."""
    estimated_runtime = inputs.window_span * arrival_scale
    sample_size_after_downsample, estimated_runtime_after_downsample = (
        _sample_size_and_estimated_runtime_at_stride(
            stride, inputs.sample_size, estimated_runtime
        )
    )
    fill_after = stride > 1
    return WindowPolicy(
        reason=reason,
        arrival_scale=arrival_scale,
        stride=stride,
        estimated_runtime=estimated_runtime,
        arrival_scale_after_downsample=(
            arrival_scale / stride if fill_after else None
        ),
        estimated_runtime_after_downsample=(
            estimated_runtime_after_downsample if fill_after else None
        ),
        sample_size_after_downsample=(
            sample_size_after_downsample if fill_after else None
        ),
    )


def _choose_stride(
    inputs: PolicyInputs,
    estimated_runtime: float,
) -> tuple[int, int, float]:
    """Smallest stride (keep every stride-th) that meets max_runtime and min_sample_size."""
    stride_needed = math.ceil(estimated_runtime / inputs.max_runtime)
    stride = max(1, stride_needed)
    sample_size_after_downsample, estimated_runtime_after_downsample = (
        _sample_size_and_estimated_runtime_at_stride(
            stride, inputs.sample_size, estimated_runtime
        )
    )
    while sample_size_after_downsample < inputs.min_sample_size and stride > 1:
        stride -= 1
        sample_size_after_downsample, estimated_runtime_after_downsample = (
            _sample_size_and_estimated_runtime_at_stride(
                stride, inputs.sample_size, estimated_runtime
            )
        )
    return stride, sample_size_after_downsample, estimated_runtime_after_downsample


def _sample_size_and_estimated_runtime_at_stride(
    stride: int,
    sample_size: int,
    estimated_runtime: float,
) -> tuple[int, float]:
    """sample_size_after_downsample and estimated_runtime at a given stride (rows[::stride])."""
    return len(range(sample_size)[::stride]), estimated_runtime / stride


def _need_longer_window(inputs: PolicyInputs, arrival_scale: float) -> WindowPolicy:
    """Fewer than min_sample_size requests. See PolicyInputs / WindowPolicy."""
    return _fill_policy(
        inputs,
        reason=PolicyReason.NEED_LONGER_RAW_WINDOW,
        arrival_scale=arrival_scale,
        stride=1,
    )


def _fit_without_downsample(inputs: PolicyInputs, arrival_scale: float) -> WindowPolicy:
    """stride=1: estimated_runtime already fits max_runtime. See PolicyInputs / WindowPolicy."""
    reason = (
        PolicyReason.KEEP_ALL_FITS
        if inputs.window_span < 120
        else PolicyReason.SCALE_ONLY
    )
    return _fill_policy(
        inputs,
        reason=reason,
        arrival_scale=arrival_scale,
        stride=1,
    )


def _downsample_policy(
    inputs: PolicyInputs, arrival_scale: float, stride: int
) -> WindowPolicy:
    """Keep every stride-th request so estimated_runtime fits. Standard downsample stride."""
    return _fill_policy(
        inputs,
        reason=PolicyReason.DOWNSAMPLE,
        arrival_scale=arrival_scale,
        stride=stride,
    )


def _shrink_min_sample_size(
    inputs: PolicyInputs, arrival_scale: float, stride: int
) -> WindowPolicy:
    """Cannot meet both max_runtime and min_sample_size. See PolicyInputs / WindowPolicy."""
    return _fill_policy(
        inputs,
        reason=PolicyReason.SHRINK_MIN_SAMPLE_SIZE,
        arrival_scale=arrival_scale,
        stride=stride,
    )


def _explain(policy: WindowPolicy, inputs: PolicyInputs) -> str:
    """Human sentence for profile explanation / stdout. Not a WindowPolicy field."""
    if policy.reason == PolicyReason.NEED_LONGER_RAW_WINDOW:
        return (
            "window has {} post-max_model_len requests < min_sample_size={}; "
            "widen the raw window before scaling inter-arrivals or downsampling"
        ).format(inputs.sample_size, inputs.min_sample_size)
    if policy.reason in (PolicyReason.KEEP_ALL_FITS, PolicyReason.SCALE_ONLY):
        return "estimated_runtime {:.1f}s <= max_runtime {:.1f}s".format(
            policy.estimated_runtime, inputs.max_runtime
        )
    if policy.reason == PolicyReason.DOWNSAMPLE:
        return (
            "scale-only estimated_runtime {:.1f}s > max_runtime; "
            "smallest stride={} gives estimated_runtime {:.1f}s "
            "and sample_size_after_downsample={}"
        ).format(
            policy.estimated_runtime,
            policy.stride,
            policy.estimated_runtime_after_downsample,
            policy.sample_size_after_downsample,
        )
    return (
        "downsample cannot meet both max_runtime and min_sample_size; "
        "last resort: shrink min_sample_size "
        "and raise the delta_min_rule accordingly"
    )


def _window_span(window: RequestWindow) -> float:
    """Seconds from first to last arrival in this window. Empty or one row -> 0."""
    if len(window.rows) < 2:
        return 0.0
    return (window.rows[-1].arrival_time - window.rows[0].arrival_time).total_seconds()


def _filter_max_model_len(
    rows: list[RecordedRequest],
    max_model_len: int,
) -> tuple[list[RecordedRequest], int]:
    """Drop requests the model cannot run: empty prompt, or prompt+output longer than max_model_len.

    Same rule as when writing the timed_trace jsonl, so sample_size here matches what
    timed_trace will keep.
    """
    after_fit = [r for r in rows if fits_max_model_len(r, max_model_len)]
    return after_fit, len(rows) - len(after_fit)


def _apply_arrival_scale(pol: dict) -> float:
    """timed_trace arrival_scale: after stride if stride > 1, else arrival_scale.

    Do not pass the pre-stride arrival_scale together with stride
    (that applies stride twice).
    """
    return pol.get("arrival_scale_after_downsample", pol["arrival_scale"])


def _apply_stride(pol: dict) -> int:
    """timed_trace stride: keep every stride-th request (1 = keep all)."""
    return int(pol.get("stride", 1))


def apply_knobs(policy: WindowPolicy) -> dict:
    """apply block for timed_trace. Do not double-apply stride."""
    pol = policy.to_dict()
    return {
        "arrival_scale": _apply_arrival_scale(pol),
        "stride": _apply_stride(pol),
    }


def _iso(ts: datetime | None) -> str | None:
    """ISO-8601 or None for profile start/end."""
    return None if ts is None else ts.isoformat()


def build_profile(
    window: RequestWindow,
    *,
    csv_path: Path,
    start: datetime | None,
    end: datetime | None,
    max_model_len: int | None,
    min_sample_size: int,
    max_runtime: float,
    guessed_rate: float | None = None,
) -> dict:
    """Replay profile from a RequestWindow and budget knobs.

    Reads argparse-equivalent knobs into PolicyInputs; recommend_policy chooses
    WindowPolicy. The human does not construct WindowPolicy.
    """
    rows = window.rows
    after_fit = rows
    if max_model_len is not None:
        after_fit, _n_dropped = _filter_max_model_len(rows, max_model_len)
    window_span = _window_span(window)
    sample_size = len(after_fit)
    inputs = PolicyInputs(
        window_span=window_span,
        sample_size=sample_size,
        guessed_rate=guessed_rate,
        max_runtime=max_runtime,
        min_sample_size=min_sample_size,
    )
    policy = recommend_policy(inputs)
    apply = apply_knobs(policy)
    if max_model_len is not None:
        apply["max_model_len"] = max_model_len
    return {
        "csv": str(csv_path),
        "start": _iso(start),
        "end": _iso(end),
        "inputs": {
            "guessed_rate": guessed_rate,
            "max_runtime": max_runtime,
            "min_sample_size": min_sample_size,
            "window_span": window_span,
            "sample_size": sample_size,
            "max_model_len": max_model_len,
        },
        "policy": policy.to_dict(),
        "apply": apply,
        "explanation": _explain(policy, inputs),
    }


def write_profile(path: Path, profile: dict) -> None:
    """Write one JSON replay profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n")


def load_profile(path: Path) -> dict:
    """Read a replay profile JSON."""
    return json.loads(path.read_text())


def profile_lines(profile: dict) -> str:
    """Short stdout: reason, apply knobs, sample_size, runtime vs budget, explanation."""
    apply = profile["apply"]
    policy = profile["policy"]
    sample_size = policy.get(
        "sample_size_after_downsample", profile["inputs"]["sample_size"]
    )
    estimated_runtime = policy.get(
        "estimated_runtime_after_downsample", policy["estimated_runtime"]
    )
    return "\n".join(
        [
            "reason={}".format(policy["reason"]),
            "--arrival-scale={:.4f} --stride={}".format(
                apply["arrival_scale"], apply["stride"]
            ),
            "sample_size={}".format(sample_size),
            "estimated_runtime={:.1f} max_runtime={:.1f}".format(
                estimated_runtime, profile["inputs"]["max_runtime"]
            ),
            profile["explanation"],
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags. You pass window + time budget; the tool chooses arrival_scale and stride."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    you = p.add_argument_group("you pass (window + time budget)")
    you.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Azure Conversation CSV (which log). Default: %(default)s",
    )
    you.add_argument("--download", action="store_true")
    you.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "ISO timestamp (inclusive). Use with --end for a specific slice "
            "once you know it. Not required with --window-sec alone "
            "(that uses the CSV's first TIMESTAMP)."
        ),
    )
    you.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help=(
            "Raw window length in seconds. Alone (no --start): first N seconds "
            "of the CSV (smoke path; uses the file's first TIMESTAMP as start). "
            "With --start: that start plus N seconds. "
            "Example once you know a slice: --start/--end from the Azure log, "
            "not a magic default."
        ),
    )
    you.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "ISO timestamp (exclusive). Specific log slice once you know it. "
            "Do not pass with --window-sec."
        ),
    )
    you.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N kept (output_length>0).",
    )
    you.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Optional. Model prompt+output limit; you may not know GPU fit yet. "
            "Stored on the profile for timed_trace --profile."
        ),
    )
    you.add_argument(
        "--min-sample-size",
        type=int,
        default=1000,
        help=(
            "Floor after stride so the mix isn't tiny. Do not keep every "
            "stride-th below this request count. Default 1000."
        ),
    )
    you.add_argument(
        "--max-runtime",
        type=float,
        default=1200.0,
        help=(
            "GPU seconds this replay may take. Default 1200 (20 min) is a smoke. "
            "For a real session, set to rental length minus slack (e.g. 3600)."
        ),
    )
    you.add_argument(
        "--guessed-rate",
        type=float,
        default=None,
        help=(
            "Optional. Bootstrap offered req/s after a first GPU run; not "
            "something you guess from thin air, not the knee. Omit for "
            "time-fit only: choose arrival_scale / stride so estimated_runtime "
            "fits max_runtime, without targeting an offered QPS."
        ),
    )
    you.add_argument(
        "--out",
        "--profile-out",
        type=Path,
        required=True,
        dest="out",
        help="Write JSON replay profile for timed_trace --profile.",
    )
    args = p.parse_args(argv)
    try:
        check_window_args(args.start, args.end, args.window_sec)
    except ValueError as err:
        p.error(str(err))
    return args


def main(argv: list[str] | None = None) -> int:
    """Parse args into PolicyInputs, load window, write profile, print short stdout."""
    args = parse_args(argv)
    csv_path = download_azure_csv(args.csv) if args.download else args.csv
    if not csv_path.exists():
        print(
            "missing {}; pass --download or --csv".format(csv_path),
            file=sys.stderr,
        )
        return 2

    try:
        start, end = resolve_window_bounds(
            csv_path, args.start, args.end, args.window_sec
        )
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    window = from_csv(csv_path, start, end, args.max_rows)
    if not window.rows:
        print("no rows in window after dropping GeneratedTokens==0", file=sys.stderr)
        return 1

    profile = build_profile(
        window,
        csv_path=csv_path,
        start=start,
        end=end,
        max_model_len=args.max_model_len,
        min_sample_size=args.min_sample_size,
        max_runtime=args.max_runtime,
        guessed_rate=args.guessed_rate,
    )
    print(profile_lines(profile))
    write_profile(args.out, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
