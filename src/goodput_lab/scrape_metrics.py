#!/usr/bin/env python3
"""Poll vLLM /metrics, write JSONL, until a stop file exists or max seconds elapse."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

# Convert ms to seconds for sleep, or seconds to ms for timestamp.
MILLISECONDS_PER_SECOND = 1000

# Allowed scrape interval, milliseconds. Outside this band the loop refuses to start.
MIN_INTERVAL_MS = 100
MAX_INTERVAL_MS = 250
DEFAULT_INTERVAL_MS = 200
# urllib timeout for each GET, seconds.
DEFAULT_TIMEOUT = 2.0


def from_url(url: str, timeout: float) -> dict:
    """GET Prometheus text from url (typically vLLM /metrics) into one JSONL record.

    timeout: urllib timeout in seconds. Stamps unix milliseconds (int) at GET
    start, the scrape time Prometheus would attach at ingest. vLLM /metrics
    usually has no sample timestamps. Success is
    {"timestamp": <int ms>, "body": <UTF-8 text>}. Fetch failure is
    {"timestamp": <int ms>, "error": <string>}; no body. JSONL line order is
    the scrape sequence. Monotonic time is for interval pacing only; it is
    not written on the record.
    """
    timestamp = int(time.time() * MILLISECONDS_PER_SECOND)
    try:
        return {"timestamp": timestamp, "body": _fetch(url, timeout)}
    except (urllib.error.URLError, TimeoutError) as err:
        return {"timestamp": timestamp, "error": str(err)}


def parse_prometheus(text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Map exposition families to gauges/counters keyed like name{labels}."""
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        family_type = (family.type or "").lower()
        for sample in family.samples:
            if sample.name.endswith("_bucket"):
                continue
            key = _series_key(sample.name, sample.labels or {})
            value = float(sample.value)
            if _is_counter_sample(family_type, sample.name):
                counters[key] = value
            else:
                gauges[key] = value
    return gauges, counters


def _fetch(url: str, timeout: float) -> str:
    """GET url as UTF-8 text. timeout is urllib seconds."""
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _series_key(name: str, labels: dict[str, str]) -> str:
    """Prometheus series key: name, or name{k="v",...} when labels are present."""
    if not labels:
        return name
    joined = ",".join('{}="{}"'.format(k, v) for k, v in labels.items())
    return "{}{{{}}}".format(name, joined)


def _is_counter_sample(family_type: str, sample_name: str) -> bool:
    """True for counter families and _total / _sum / _count samples."""
    return (
        family_type == "counter"
        or sample_name.endswith("_total")
        or sample_name.endswith("_sum")
        or sample_name.endswith("_count")
    )


def should_stop(
    stop_file: Path | None, max_seconds: float | None, t_start: float
) -> bool:
    """True when stop_file exists or monotonic time since t_start exceeds max_seconds.

    stop_file: path the bench runner creates when the run is done. max_seconds:
    optional wall budget when there is no stop file. t_start is monotonic.
    """
    if stop_file is not None and stop_file.exists():
        return True
    if max_seconds is not None and (time.monotonic() - t_start) >= max_seconds:
        return True
    return False


def sleep_remainder(interval_ms: int, started: float) -> None:
    """Sleep the unused part of interval_ms after a scrape that started at started.

    started is monotonic seconds. interval_ms converts to seconds only for
    time.sleep. If the GET already used the whole interval, do not sleep.
    Pacing uses monotonic time only; it is not written on the JSONL record.
    """
    sleep_for = interval_ms / MILLISECONDS_PER_SECOND - (time.monotonic() - started)
    if sleep_for > 0:
        time.sleep(sleep_for)


def parse_args() -> argparse.Namespace:
    """CLI flags. Interval is milliseconds in the freeze band; timeout is urllib seconds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="{}-{} ms.".format(MIN_INTERVAL_MS, MAX_INTERVAL_MS),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Exit when this file exists (run.sh creates it after bench).",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser.parse_args()


def main() -> int:
    """Poll url on the interval; write JSONL until stop_file exists or max_seconds elapse."""
    args = parse_args()
    if not (MIN_INTERVAL_MS <= args.interval_ms <= MAX_INTERVAL_MS):
        print(
            "interval-ms={} outside {}-{} ms freeze band".format(
                args.interval_ms, MIN_INTERVAL_MS, MAX_INTERVAL_MS
            ),
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    n_ok = 0
    n_fail = 0
    with args.out.open("w") as out:
        while not should_stop(args.stop_file, args.max_seconds, t_start):
            started = time.monotonic()
            record = from_url(args.url, args.timeout)
            if "body" in record:
                n_ok += 1
            else:
                n_fail += 1
            out.write(json.dumps(record) + "\n")
            out.flush()
            sleep_remainder(args.interval_ms, started)
    print("scrape: wrote {} ok={} fail={}".format(args.out, n_ok, n_fail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
