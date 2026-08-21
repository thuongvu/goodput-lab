#!/usr/bin/env python3
"""
Scrape: scrape_url GETs /metrics; CLI writes jsonl. Started by run.sh.
Parse: local metrics.jsonl -> Scrape / WindowStats. Imported by results.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

MILLISECONDS_PER_SECOND = 1000

RUNNING_SERIES = "vllm:num_requests_running"
KV_SERIES = "vllm:kv_cache_usage_perc"
PREEMPTION_SERIES = "vllm:num_preemptions_total"
_PIN_SERIES = (RUNNING_SERIES, KV_SERIES, PREEMPTION_SERIES)


@dataclass(frozen=True)
class Scrape:
    """One metrics.jsonl row that had a Prometheus body, parsed into gauges and counters."""

    unix_ms: int
    gauges: dict[str, float]
    counters: dict[str, float]


# Scrape
# scrape_url GETs /metrics; CLI writes jsonl. Started by run.sh.
_MIN_INTERVAL_MS = 100
_MAX_INTERVAL_MS = 250
_DEFAULT_INTERVAL_MS = 200
# urllib timeout for each GET, seconds.
_DEFAULT_TIMEOUT = 2.0


def scrape_url(url: str, timeout: float) -> dict:
    """GET /metrics; return unix_ms (wall clock at GET start) plus body or error.

    CLI writes one jsonl line per call.
    """
    timestamp = int(time.time() * MILLISECONDS_PER_SECOND)
    try:
        return {"timestamp": timestamp, "body": _fetch(url, timeout)}
    except (urllib.error.URLError, TimeoutError) as err:
        return {"timestamp": timestamp, "error": str(err)}


def _fetch(url: str, timeout: float) -> str:
    """GET url as UTF-8 text."""
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _should_stop(
    stop_file: Path | None, max_seconds: float | None, loop_started: float
) -> bool:
    """True when stop_file exists or monotonic time since loop_started exceeds max_seconds.

    stop_file: path the bench runner creates when the run is done. 
    max_seconds: optional max run time. loop_started is monotonic seconds from time.monotonic
    when the scrape loop began.
    """
    if stop_file is not None and stop_file.exists():
        return True
    if max_seconds is not None and (time.monotonic() - loop_started) >= max_seconds:
        return True
    return False


def _sleep_remainder(interval_ms: int, started: float) -> None:
    """After a GET that began at started (monotonic), sleep so the next GET starts about interval_ms later.

    If the GET already took that long, do not sleep. interval_ms is converted to
    seconds only for time.sleep.
    """
    sleep_for = interval_ms / MILLISECONDS_PER_SECOND - (time.monotonic() - started)
    if sleep_for > 0:
        time.sleep(sleep_for)


# Parse
# Local metrics.jsonl -> Scrape / WindowStats. Imported by results.py.
@dataclass(frozen=True)
class WindowStats:
    """Prometheus occupancy, KV fill, and preemption over a time window.

    running is in-flight request count. kv is KV cache fill fraction.
    preemption_delta is the increase in the preemption counter over the window.
    """

    running_median: float | None
    running_p90: float | None
    kv_median: float | None
    kv_p90: float | None
    preemption_delta: float | None


def from_jsonl(path: Path) -> list[Scrape]:
    """Read a local metrics.jsonl path into Scrape rows."""
    scrapes = _parse_scrapes(_load_records(path))
    _require_series(scrapes)
    return scrapes


def window_stats(
    scrapes: list[Scrape],
    first_send_unix_ms: int,
    start: float,
    end: float,
) -> WindowStats:
    """Scrapes in [start, end) seconds after first send (first_send_unix_ms aligns scrape Unix ms)."""
    running: list[float] = []
    kv: list[float] = []
    preemption: list[float] = []
    for scrape in scrapes:
        relative = (scrape.unix_ms - first_send_unix_ms) / MILLISECONDS_PER_SECOND
        if relative < start:
            continue
        if relative >= end:
            continue
        running_value = _lookup(scrape, RUNNING_SERIES)
        if running_value is not None:
            running.append(running_value)
        kv_value = _lookup(scrape, KV_SERIES)
        if kv_value is not None:
            kv.append(kv_value)
        preemption_value = _lookup(scrape, PREEMPTION_SERIES)
        if preemption_value is not None:
            preemption.append(preemption_value)
    return WindowStats(
        running_median=_percentile(running, 50),
        running_p90=_percentile(running, 90),
        kv_median=_percentile(kv, 50),
        kv_p90=_percentile(kv, 90),
        preemption_delta=_preemption_delta(preemption),
    )


def running_count(scrape: Scrape) -> float | None:
    """In-flight request count for this scrape."""
    return _lookup(scrape, RUNNING_SERIES)


def _load_records(path: Path) -> list[dict]:
    """Parse metrics.jsonl scrape records from a local path."""
    records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_scrapes(records: list[dict]) -> list[Scrape]:
    """Unix-ms timestamp plus gauges/counters for each scrape that has a body."""
    scrapes: list[Scrape] = []
    for record in records:
        body = record.get("body")
        if not body:
            continue
        gauges, counters = _parse_prometheus(body)
        scrapes.append(
            Scrape(unix_ms=int(record["timestamp"]), gauges=gauges, counters=counters)
        )
    return scrapes


def _parse_prometheus(text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Map parsed /metrics samples to gauges/counters keyed like name{labels}."""
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    for metric in text_string_to_metric_families(text):
        metric_type = (metric.type or "").lower()
        for sample in metric.samples:
            if sample.name.endswith("_bucket"):
                continue
            key = _series_key(sample.name, sample.labels or {})
            value = float(sample.value)
            if _is_counter_sample(metric_type, sample.name):
                counters[key] = value
            else:
                gauges[key] = value
    return gauges, counters


def _series_key(name: str, labels: dict[str, str]) -> str:
    """Prometheus series key: name, or name{k="v",...}."""
    if not labels:
        return name
    joined = ",".join('{}="{}"'.format(k, v) for k, v in labels.items())
    return "{}{{{}}}".format(name, joined)


def _is_counter_sample(family_type: str, sample_name: str) -> bool:
    """True for counter metrics and _total / _sum / _count samples."""
    return (
        family_type == "counter"
        or sample_name.endswith("_total")
        or sample_name.endswith("_sum")
        or sample_name.endswith("_count")
    )


def _require_series(scrapes: list[Scrape]) -> None:
    """Error if a required series never appears in any scrape body."""
    for name in _PIN_SERIES:
        if not any(_lookup(scrape, name) is not None for scrape in scrapes):
            raise ValueError(
                "missing Prometheus series {} in metrics.jsonl".format(name)
            )


def _lookup(scrape: Scrape, name: str) -> float | None:
    """Exact series name, or the sum of labeled name{...} metrics."""
    for mapping in (scrape.gauges, scrape.counters):
        if name in mapping:
            return mapping[name]
        prefix = name + "{"
        matched = [value for key, value in mapping.items() if key.startswith(prefix)]
        if matched:
            return sum(matched)
    return None


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile of the given values (running count, KV fill). Empty -> None."""
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


def _preemption_delta(samples: list[float]) -> float | None:
    """Last minus first counter value in the window. Empty -> None."""
    if not samples:
        return None
    return samples[-1] - samples[0]


# CLI
def parse_args() -> argparse.Namespace:
    """CLI flags for the live scrape writer. Interval is milliseconds"""
    parser = argparse.ArgumentParser(
        description=(
            "Poll vLLM /metrics, write JSONL, until a stop file exists or max seconds elapse."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=_DEFAULT_INTERVAL_MS,
        help="scrape interval, {}-{} ms.".format(_MIN_INTERVAL_MS, _MAX_INTERVAL_MS),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Exit when this file exists (run.sh creates it after bench).",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)
    return parser.parse_args()


def main() -> int:
    """Poll url on the scrape interval; write raw bodies to metrics.jsonl."""
    args = parse_args()
    if not (_MIN_INTERVAL_MS <= args.interval_ms <= _MAX_INTERVAL_MS):
        print(
            "interval-ms={} out of allowed scrape interval ({}-{}ms)".format(
                args.interval_ms, _MIN_INTERVAL_MS, _MAX_INTERVAL_MS
            ),
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    loop_started = time.monotonic()
    ok_scrapes = 0
    failed_scrapes = 0
    with args.out.open("w") as out:
        while not _should_stop(args.stop_file, args.max_seconds, loop_started):
            started = time.monotonic()
            record = scrape_url(args.url, args.timeout)
            if "body" in record:
                ok_scrapes += 1
            else:
                failed_scrapes += 1
            out.write(json.dumps(record) + "\n")
            out.flush()
            _sleep_remainder(args.interval_ms, started)
    print("scrape: wrote {} ok={} fail={}".format(args.out, ok_scrapes, failed_scrapes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
