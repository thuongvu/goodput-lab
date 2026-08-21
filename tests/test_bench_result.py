#!/usr/bin/env python3
"""Local tests for bench_result. Stdlib unittest; runs locally."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from goodput_lab.bench_result import from_json


class TestFromJson(unittest.TestCase):
    """from_json reads per-request arrays from bench.json."""

    def test_whole_file_json(self) -> None:
        """Whole-file JSON yields ttfts/itls/input_lens lists."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(
                json.dumps(
                    {
                        "input_lens": [16, 16],
                        "output_lens": [8, 8],
                        "ttfts": [0.01, 0.02],
                        "itls": [[0.002], [0.003]],
                    }
                )
            )
            bench = from_json(path)
            self.assertEqual(bench.input_lens, [16, 16])
            self.assertEqual(bench.ttfts, [0.01, 0.02])
            self.assertEqual(bench.itls, [[0.002], [0.003]])
            self.assertIsNone(bench.queue_times)
            self.assertIsNone(bench.start_times)
            self.assertIsNone(bench.duration)

    def test_start_times_and_duration(self) -> None:
        """Optional start_times and duration are kept when present."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(
                json.dumps(
                    {
                        "input_lens": [16],
                        "output_lens": [8],
                        "ttfts": [0.01],
                        "itls": [[0.002]],
                        "start_times": [12.5],
                        "duration": 3.25,
                    }
                )
            )
            bench = from_json(path)
            self.assertEqual(bench.start_times, [12.5])
            self.assertEqual(bench.duration, 3.25)

    def test_missing_ttfts_errors(self) -> None:
        """Missing ttfts raises KeyError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(json.dumps({"input_lens": [16], "itls": [[0.1]]}))
            with self.assertRaises(KeyError) as ctx:
                from_json(path)
            self.assertIn("ttfts", str(ctx.exception))

    def test_null_required_arrays_are_join_errors(self) -> None:
        """JSON null ttfts, itls, or input_lens raises ValueError."""
        base = {
            "input_lens": [16],
            "output_lens": [8],
            "ttfts": [0.01],
            "itls": [[0.002]],
        }
        for key in ("ttfts", "itls", "input_lens"):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "bench.json"
                    payload = dict(base)
                    payload[key] = None
                    path.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError) as ctx:
                        from_json(path)
                    self.assertIn(key, str(ctx.exception))
                    self.assertNotIsInstance(ctx.exception, TypeError)

    def test_ttfts_not_a_list_is_join_error(self) -> None:
        """A non-list ttfts value raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(
                json.dumps(
                    {
                        "input_lens": [16],
                        "output_lens": [8],
                        "ttfts": 0.01,
                        "itls": [[0.002]],
                    }
                )
            )
            with self.assertRaises(ValueError) as ctx:
                from_json(path)
            self.assertIn("ttfts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
