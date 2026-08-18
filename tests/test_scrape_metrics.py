#!/usr/bin/env python3
"""Unit tests for scrape_metrics: from_url records, should_stop, parse_prometheus."""

from __future__ import annotations

import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from goodput_lab.scrape_metrics import from_url, parse_prometheus, should_stop

PROM_BODY = "# TYPE foo gauge\nfoo 1.5\n"


class TestFromUrl(unittest.TestCase):
    def test_success_record_keys(self) -> None:
        with patch(
            "goodput_lab.scrape_metrics._fetch", return_value=PROM_BODY
        ) as fetch:
            with patch(
                "goodput_lab.scrape_metrics.time.time", return_value=1700000000.5
            ):
                record = from_url("http://127.0.0.1:8000/metrics", 2.0)
        fetch.assert_called_once_with("http://127.0.0.1:8000/metrics", 2.0)
        self.assertEqual(set(record), {"timestamp", "body"})
        self.assertIsInstance(record["timestamp"], int)
        self.assertEqual(record["timestamp"], 1700000000500)
        self.assertEqual(record["body"], PROM_BODY)
        self.assertNotIn("error", record)

    def test_failure_record_has_error_no_body(self) -> None:
        err = urllib.error.URLError("refused")
        with patch("goodput_lab.scrape_metrics._fetch", side_effect=err):
            with patch(
                "goodput_lab.scrape_metrics.time.time", return_value=1700000000.5
            ):
                record = from_url("http://127.0.0.1:8000/metrics", 2.0)
        self.assertEqual(set(record), {"timestamp", "error"})
        self.assertIsInstance(record["timestamp"], int)
        self.assertEqual(record["timestamp"], 1700000000500)
        self.assertEqual(record["error"], str(err))
        self.assertNotIn("body", record)


class TestShouldStop(unittest.TestCase):
    def test_stop_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop"
            stop.write_text("")
            self.assertTrue(should_stop(stop, None, time.monotonic()))

    def test_max_seconds_elapsed(self) -> None:
        t_start = time.monotonic() - 10.0
        self.assertTrue(should_stop(None, 1.0, t_start))

    def test_neither_stop_file_nor_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop"
            self.assertFalse(should_stop(stop, 100.0, time.monotonic()))
            self.assertFalse(should_stop(None, None, time.monotonic()))


class TestParsePrometheus(unittest.TestCase):
    def test_gauge_and_counter_snippet(self) -> None:
        text = (
            "# TYPE foo gauge\n"
            "foo 1.5\n"
            "# TYPE requests_total counter\n"
            "requests_total 10\n"
        )
        gauges, counters = parse_prometheus(text)
        self.assertEqual(gauges["foo"], 1.5)
        self.assertEqual(counters["requests_total"], 10.0)
        self.assertNotIn("foo", counters)
        self.assertNotIn("requests_total", gauges)


if __name__ == "__main__":
    unittest.main()
