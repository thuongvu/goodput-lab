#!/usr/bin/env python3
"""Local tests for metrics_jsonl. Stdlib unittest; runs locally."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from goodput_lab.metrics_jsonl import (
    KV_SERIES,
    PREEMPTION_SERIES,
    RUNNING_SERIES,
    Scrape,
    _parse_prometheus,
    _should_stop,
    from_jsonl,
    scrape_url,
    window_stats,
)

PROM_BODY = "# TYPE foo gauge\nfoo 1.5\n"


def _prom_body(running: float, kv: float, preemptions: float) -> str:
    """Tiny Prometheus text with running, KV, and preemption samples."""
    return (
        "# TYPE vllm:num_requests_running gauge\n"
        "vllm:num_requests_running {}\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        "vllm:kv_cache_usage_perc {}\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        "vllm:num_preemptions_total {}\n"
    ).format(running, kv, preemptions)


def _labeled_prom_body() -> str:
    """Prometheus text with engine/model labels on the pin series."""
    return (
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0\n'
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        'vllm:kv_cache_usage_perc{engine="0"} 0.1\n'
        "# TYPE vllm:num_preemptions_total counter\n"
        'vllm:num_preemptions_total{engine="0"} 3.0\n'
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


class TestFromJsonl(unittest.TestCase):
    """from_jsonl requires pin series; labeled names still match."""

    def test_missing_series_errors(self) -> None:
        """A body without the KV series raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            body = (
                "# TYPE vllm:num_requests_running gauge\n"
                "vllm:num_requests_running 1.0\n"
                "# TYPE vllm:num_preemptions_total counter\n"
                "vllm:num_preemptions_total 0.0\n"
            )
            _write_jsonl(path, [{"timestamp": 1_000_000, "body": body}])
            with self.assertRaises(ValueError) as ctx:
                from_jsonl(path)
            self.assertIn(KV_SERIES, str(ctx.exception))

    def test_labeled_series_match_pin_names(self) -> None:
        """Prefix lookup sums name{engine=...} children as the pin series."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            _write_jsonl(
                path,
                [{"timestamp": 1_000_000, "body": _labeled_prom_body()}],
            )
            scrapes = from_jsonl(path)
            stats = window_stats(scrapes, 1_000_000, 0.0, 1.0)
            self.assertEqual(stats.running_median, 2.0)
            self.assertAlmostEqual(stats.kv_median, 0.1)
            self.assertEqual(stats.preemption_delta, 0.0)


class TestWindowStats(unittest.TestCase):
    """window_stats slices relative to first_send_unix_ms."""

    def test_healthy_window_quantiles_and_preemption_delta(self) -> None:
        """Median running/KV and last-minus-first preemption in [0, 10)."""
        scrapes = [
            Scrape(
                unix_ms=1_000_000,
                gauges={RUNNING_SERIES: 0.0, KV_SERIES: 0.0},
                counters={PREEMPTION_SERIES: 2.0},
            ),
            Scrape(
                unix_ms=1_001_000,
                gauges={RUNNING_SERIES: 1.0, KV_SERIES: 0.1},
                counters={PREEMPTION_SERIES: 2.0},
            ),
            Scrape(
                unix_ms=1_002_000,
                gauges={RUNNING_SERIES: 3.0, KV_SERIES: 0.2},
                counters={PREEMPTION_SERIES: 5.0},
            ),
            Scrape(
                unix_ms=1_011_000,
                gauges={RUNNING_SERIES: 8.0, KV_SERIES: 0.5},
                counters={PREEMPTION_SERIES: 5.0},
            ),
        ]
        stats = window_stats(scrapes, 1_001_000, 0.0, 10.0)
        self.assertEqual(stats.running_median, 2.0)
        self.assertAlmostEqual(stats.kv_median, 0.15)
        self.assertEqual(stats.preemption_delta, 3.0)


class TestScrapeUrl(unittest.TestCase):
    """scrape_url stamps unix ms and keeps body or error, never both."""

    def test_success_record_keys(self) -> None:
        """Success record is timestamp plus body."""
        with patch(
            "goodput_lab.metrics_jsonl._fetch", return_value=PROM_BODY
        ) as fetch:
            with patch(
                "goodput_lab.metrics_jsonl.time.time", return_value=1700000000.5
            ):
                record = scrape_url("http://127.0.0.1:8000/metrics", 2.0)
        fetch.assert_called_once_with("http://127.0.0.1:8000/metrics", 2.0)
        self.assertEqual(set(record), {"timestamp", "body"})
        self.assertIsInstance(record["timestamp"], int)
        self.assertEqual(record["timestamp"], 1700000000500)
        self.assertEqual(record["body"], PROM_BODY)
        self.assertNotIn("error", record)

    def test_failure_record_has_error_no_body(self) -> None:
        """Fetch failure is timestamp plus error."""
        err = urllib.error.URLError("refused")
        with patch("goodput_lab.metrics_jsonl._fetch", side_effect=err):
            with patch(
                "goodput_lab.metrics_jsonl.time.time", return_value=1700000000.5
            ):
                record = scrape_url("http://127.0.0.1:8000/metrics", 2.0)
        self.assertEqual(set(record), {"timestamp", "error"})
        self.assertIsInstance(record["timestamp"], int)
        self.assertEqual(record["timestamp"], 1700000000500)
        self.assertEqual(record["error"], str(err))
        self.assertNotIn("body", record)


class TestShouldStop(unittest.TestCase):
    """Stop when the stop file exists or max_seconds elapses."""

    def test_stop_file_exists(self) -> None:
        """An existing stop file ends the loop."""
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop"
            stop.write_text("")
            self.assertTrue(_should_stop(stop, None, time.monotonic()))

    def test_max_seconds_elapsed(self) -> None:
        """Elapsed monotonic time past max_seconds ends the loop."""
        loop_started = time.monotonic() - 10.0
        self.assertTrue(_should_stop(None, 1.0, loop_started))

    def test_neither_stop_file_nor_max_seconds(self) -> None:
        """An absent stop file with remaining max_seconds keeps polling."""
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop"
            self.assertFalse(_should_stop(stop, 100.0, time.monotonic()))
            self.assertFalse(_should_stop(None, None, time.monotonic()))


class TestParsePrometheus(unittest.TestCase):
    """_parse_prometheus splits gauges from counters."""

    def test_gauge_and_counter_snippet(self) -> None:
        """A gauge stays in gauges; a _total sample stays in counters."""
        text = (
            "# TYPE foo gauge\n"
            "foo 1.5\n"
            "# TYPE requests_total counter\n"
            "requests_total 10\n"
        )
        gauges, counters = _parse_prometheus(text)
        self.assertEqual(gauges["foo"], 1.5)
        self.assertEqual(counters["requests_total"], 10.0)
        self.assertNotIn("foo", counters)
        self.assertNotIn("requests_total", gauges)


if __name__ == "__main__":
    unittest.main()
