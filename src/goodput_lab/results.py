#!/usr/bin/env python3
"""Join a run's request jsonl, vLLM bench.json, and metrics.jsonl into results.json,
with nested stats per phase. Bench latencies join by row index; metrics.jsonl joins by
wall clock.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from goodput_lab.bench_result import BenchResult, from_json as bench_from_json
from goodput_lab.generate_workload import Phase
from goodput_lab.metrics_jsonl import (
    MILLISECONDS_PER_SECOND,
    RUNNING_SERIES,
    Scrape,
    WindowStats,
    from_jsonl as metrics_from_jsonl,
    running_count,
    window_stats,
)
from goodput_lab.serve_settings import from_run_dir as serve_settings_from_run_dir
from goodput_lab.serve_settings import write as write_serve_settings

RESULTS_FILENAME = "results.json"


# RESULTS
# Lab results.json as a record. Nested PhaseStats per phase we sent.


@dataclass(frozen=True)
class PhaseStats:
    """Percentiles for one phase. Bench arrays are seconds; latencies here are milliseconds."""

    ttft_p50: float | None
    ttft_p90: float | None
    ttft_p99: float | None
    itl_p50: float | None
    itl_p90: float | None
    itl_p99: float | None
    window_stats: WindowStats
    queue_time_p50: float | None = None
    queue_time_p90: float | None = None
    queue_time_p99: float | None = None

    def to_dict(self) -> dict:
        """One phase for results.json, with window stats inlined. Omits empty queue_time keys."""
        payload = {
            "ttft_p50": self.ttft_p50,
            "ttft_p90": self.ttft_p90,
            "ttft_p99": self.ttft_p99,
            "itl_p50": self.itl_p50,
            "itl_p90": self.itl_p90,
            "itl_p99": self.itl_p99,
        }
        payload.update(asdict(self.window_stats))
        if self.queue_time_p50 is None:
            return payload
        payload["queue_time_p50"] = self.queue_time_p50
        payload["queue_time_p90"] = self.queue_time_p90
        payload["queue_time_p99"] = self.queue_time_p99
        return payload


@dataclass(frozen=True)
class Results:
    """Lab results.json as a record, with nested stats per phase."""

    by_phase: dict[str, PhaseStats]

    def to_dict(self) -> dict:
        """Stats for each phase that has rows. Keys are the jsonl phase strings."""
        return {name: stats.to_dict() for name, stats in self.by_phase.items()}


# JOIN
# Load jsonl + vLLM bench.json + metrics.jsonl; index join (ttfts[i] is jsonl row i) and wall-time join into lab Results.


def results_from_run_dir(run_dir: Path, trace: Path) -> Results:
    """Load the served jsonl, bench.json, and metrics.jsonl; join into Results."""
    _refuse_repeat_copies(run_dir)
    rows = rows_from_timed_trace(trace)
    bench = bench_from_json(run_dir / "bench.json")
    scrapes = metrics_from_jsonl(run_dir / "metrics.jsonl")
    return _results_from_rows(rows, bench, scrapes)


def _results_from_rows(
    rows: list[dict], bench: BenchResult, scrapes: list[Scrape]
) -> Results:
    """Join jsonl rows to bench latencies by index and metrics.jsonl by wall time."""
    _require_aligned(rows, bench)
    phase_names = _phase_names(rows)
    _require_healthy(phase_names)
    first_send = _first_send_unix_ms(scrapes, bench)
    windows = _phase_windows(rows, bench, phase_names)
    by_phase: dict[str, PhaseStats] = {}
    for name in phase_names:
        indices = _phase_indices(rows, name)
        start, end = windows[name]
        by_phase[name] = _phase_stats(
            indices, bench, scrapes, first_send, start, end
        )
    return Results(by_phase=by_phase)


def rows_from_timed_trace(path: Path) -> list[dict]:
    """Load nonempty jsonl lines as dicts. Same file that was served."""
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty trace {}".format(path))
    return rows


def _refuse_repeat_copies(run_dir: Path) -> None:
    """Ensures a single bench.json so metrics.jsonl lines up with that run."""
    copies = sorted(path.name for path in run_dir.glob("rep_*.json"))
    if copies:
        raise ValueError(
            "run dir has {}; summarize one bench.json run, or omit repeat copies".format(
                ", ".join(copies)
            )
        )


def _require_aligned(rows: list[dict], bench: BenchResult) -> None:
    """Ensures bench arrays and jsonl have the same number of rows."""
    n_rows = len(rows)
    if len(bench.input_lens) != n_rows:
        raise ValueError(
            "bench input_lens count {} != trace rows {}".format(
                len(bench.input_lens), n_rows
            )
        )
    if len(bench.ttfts) != n_rows:
        raise ValueError("bench ttfts missing or length does not match trace")
    if len(bench.itls) != n_rows:
        raise ValueError("bench itls missing or length does not match trace")
    if bench.queue_times is not None and len(bench.queue_times) != n_rows:
        raise ValueError("bench queue_times length does not match trace")
    if bench.start_times is not None and len(bench.start_times) != n_rows:
        raise ValueError("bench start_times length does not match trace")


# FIRST SEND
# Wall-clock origin of the jsonl: last scrape with running > 0 (in-flight), minus bench.duration.


def _first_send_unix_ms(scrapes: list[Scrape], bench: BenchResult) -> int:
    """Wall-clock Unix ms of the first send: last in-flight scrape minus bench duration."""
    elapsed = bench.duration
    if elapsed is None:
        raise ValueError("bench duration is missing; cannot set first_send_unix_ms")
    last_running = _last_running_unix_ms(scrapes)
    return int(last_running - elapsed * MILLISECONDS_PER_SECOND)


def _last_running_unix_ms(scrapes: list[Scrape]) -> int:
    """Unix ms of the last scrape that still had in-flight requests (running > 0)."""
    last = None
    for scrape in scrapes:
        value = running_count(scrape)
        if value is not None and value > 0:
            last = scrape.unix_ms
    if last is None:
        raise ValueError(
            "no scrape with {} > 0; cannot set first_send_unix_ms".format(
                RUNNING_SERIES
            )
        )
    return last


# PHASE WINDOWS
# Each phase from its first send to when that phase's last request finishes.


def _phase_names(rows: list[dict]) -> list[str]:
    """Distinct jsonl phase tags in first-appearance order."""
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = _row_phase(row)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _require_healthy(phase_names: list[str]) -> None:
    """Ensures healthy rows exist when the trace uses healthy/busy/pressure/recovery tags."""
    decode_phases = {phase.value for phase in Phase}
    present = set(phase_names)
    if present & decode_phases and "healthy" not in present:
        raise ValueError("no healthy rows in trace")


def _phase_windows(
    rows: list[dict], bench: BenchResult, phase_names: list[str]
) -> dict[str, tuple[float, float]]:
    """Scrape windows on the jsonl relative-second clock, half-open [start, end).

    Each phase starts at its first send and ends when that phase's last request finishes.
    """
    windows: dict[str, tuple[float, float]] = {}
    for name in phase_names:
        indices = _phase_indices(rows, name)
        start = float(rows[indices[0]]["timestamp"])
        last = indices[-1]
        end = float(rows[last]["timestamp"]) + _request_duration(bench, last)
        if end <= start:
            end = start + (1.0 / MILLISECONDS_PER_SECOND)
        windows[name] = (start, end)
    return windows


def _row_phase(row: dict) -> str:
    """Intention tag on a jsonl row (phase)."""
    return row["phase"]


def _request_duration(bench: BenchResult, index: int) -> float:
    """Seconds from send to last token: TTFT plus ITL samples for this bench row. Missing pieces -> 0."""
    ttft = bench.ttfts[index]
    total = 0.0 if ttft is None else float(ttft)
    samples = bench.itls[index]
    if samples is None:
        return total
    if not isinstance(samples, list):
        samples = [samples]
    for sample in samples:
        if sample is not None:
            total += float(sample)
    return total


def _phase_indices(rows: list[dict], phase: str) -> list[int]:
    """Row indexes tagged with this phase. Index join into bench arrays uses these."""
    return [
        index for index, row in enumerate(rows) if _row_phase(row) == phase
    ]


# PHASE STATS
# TTFT/ITL by index, occupancy by wall time.


def _phase_stats(
    indices: list[int],
    bench: BenchResult,
    scrapes: list[Scrape],
    first_send_unix_ms: int,
    start: float,
    end: float,
) -> PhaseStats:
    """TTFT/ITL percentiles (index join) and window stats (wall-time join) for one phase."""
    ttft_ms, itl_ms, queue_ms = _join_ttft_itl(indices, bench)
    if _empty_scrape_window(scrapes, first_send_unix_ms, start, end):
        raise ValueError(
            "no scrapes in [{}, {}) after first_send_unix_ms={}".format(
                start, end, first_send_unix_ms
            )
        )
    return PhaseStats(
        ttft_p50=_percentile(ttft_ms, 50),
        ttft_p90=_percentile(ttft_ms, 90),
        ttft_p99=_percentile(ttft_ms, 99),
        itl_p50=_percentile(itl_ms, 50),
        itl_p90=_percentile(itl_ms, 90),
        itl_p99=_percentile(itl_ms, 99),
        window_stats=window_stats(scrapes, first_send_unix_ms, start, end),
        queue_time_p50=_percentile(queue_ms, 50) if queue_ms is not None else None,
        queue_time_p90=_percentile(queue_ms, 90) if queue_ms is not None else None,
        queue_time_p99=_percentile(queue_ms, 99) if queue_ms is not None else None,
    )


def _join_ttft_itl(
    indices: list[int], bench: BenchResult
) -> tuple[list[float], list[float], list[float] | None]:
    """Bench TTFT/ITL (and optional queue) for these row indexes, in milliseconds."""
    ttfts = bench.ttfts
    itls = bench.itls
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    for index in indices:
        ttft = ttfts[index]
        if ttft is None:
            raise ValueError("null ttft at index {}".format(index))
        ttft_ms.append(float(ttft) * MILLISECONDS_PER_SECOND)
        samples = itls[index]
        if samples is None:
            raise ValueError("null itl at index {}".format(index))
        if not isinstance(samples, list):
            samples = [samples]
        for sample in samples:
            if sample is None:
                raise ValueError("null itl sample at index {}".format(index))
            itl_ms.append(float(sample) * MILLISECONDS_PER_SECOND)
    queue_times = bench.queue_times
    if queue_times is None:
        return ttft_ms, itl_ms, None
    queue_ms = [
        float(queue_times[index]) * MILLISECONDS_PER_SECOND
        for index in indices
        if queue_times[index] is not None
    ]
    return ttft_ms, itl_ms, queue_ms


def _empty_scrape_window(
    scrapes: list[Scrape], first_send_unix_ms: int, start: float, end: float
) -> bool:
    """True when no scrape wall time falls in [start, end) seconds after first_send_unix_ms."""
    for scrape in scrapes:
        relative = (scrape.unix_ms - first_send_unix_ms) / MILLISECONDS_PER_SECOND
        if start <= relative < end:
            return False
    return True


def _percentile(values: list[float], p: float) -> float | None:
    """Percentile of the given values (phase latencies).

    Rank is p/100 times (n-1). If that lands between two samples, blend them.
    """
    if not values:
        return None
    ordered = sorted(values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


# WRITE RESULTS
# results.json into the run dir. serve-settings.json is a sibling (serve_settings.py).


def write_results(run_dir: Path, results: Results) -> Path:
    """Write this run's results.json into run_dir."""
    path = run_dir / RESULTS_FILENAME
    path.write_text(json.dumps(results.to_dict(), indent=2) + "\n")
    return path


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: --run-dir and --trace. --trace must be the same jsonl that was served."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="generated timed-trace jsonl",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Join run dir + the served jsonl; write results.json, then sibling serve-settings.json."""
    args = parse_args(argv)
    try:
        results = results_from_run_dir(args.run_dir, args.trace)
        results_path = write_results(args.run_dir, results)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as err:
        print("results.json: {}".format(err), file=sys.stderr)
        return 1
    try:
        settings = serve_settings_from_run_dir(args.run_dir)
        settings_path = write_serve_settings(args.run_dir, settings)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as err:
        print("serve-settings.json: {}".format(err), file=sys.stderr)
        return 1
    print("wrote {}\nwrote {}".format(results_path, settings_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
