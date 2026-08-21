#!/usr/bin/env python3
"""Local tests for count_match: matching counts, mismatch stderr."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goodput_lab.count_match import count_nonempty_lines, from_paths, main


def _write_trace(path: Path, line_count: int) -> None:
    """Write line_count one-object jsonl rows."""
    path.write_text(
        "".join("{}\n".format(json.dumps({"i": i})) for i in range(line_count))
    )


class TestCountMatch(unittest.TestCase):
    """from_paths and CLI count-match against bench input_lens."""

    def test_matching_counts(self) -> None:
        """Equal jsonl and input_lens counts are ok."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            bench = root / "bench.json"
            _write_trace(trace, 3)
            bench.write_text(
                json.dumps({"input_lens": [1, 2, 3], "output_lens": [4, 5, 6]})
            )
            match = from_paths(trace, bench)
            self.assertTrue(match.ok)
            self.assertEqual(match.timed_trace_count, 3)
            self.assertEqual(match.input_count, 3)
            self.assertEqual(match.output_count, 3)

    def test_missing_output_lens_not_mismatch(self) -> None:
        """Missing or empty output_lens is 0 and still matches."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            bench = root / "bench.json"
            _write_trace(trace, 3)
            bench.write_text(json.dumps({"input_lens": [1, 2, 3]}))
            match = from_paths(trace, bench)
            self.assertTrue(match.ok)
            self.assertEqual(match.timed_trace_count, 3)
            self.assertEqual(match.input_count, 3)
            self.assertEqual(match.output_count, 0)
            bench.write_text(
                json.dumps({"input_lens": [1, 2, 3], "output_lens": []})
            )
            empty = from_paths(trace, bench)
            self.assertTrue(empty.ok)
            self.assertEqual(empty.output_count, 0)

    def test_mismatch_stderr(self) -> None:
        """CLI exits 1 and prints jsonl vs input_lens counts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            bench = root / "bench.json"
            _write_trace(trace, 3)
            bench.write_text(
                json.dumps({"input_lens": [1, 2], "output_lens": [4, 5]})
            )
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                code = main([str(trace), str(bench)])
            self.assertEqual(code, 1)
            err = buf.getvalue()
            self.assertIn(
                "count_match: jsonl rows 3 != bench input_lens 2", err
            )

    def test_count_nonempty_lines_skips_blank(self) -> None:
        """Blank jsonl lines are skipped; only nonempty lines count."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "trace.jsonl"
            path.write_text(
                "{}\n\n  \n{}\n".format(
                    json.dumps({"i": 0}), json.dumps({"i": 1})
                )
            )
            self.assertEqual(count_nonempty_lines(path), 2)
            bench = root / "bench.json"
            bench.write_text(
                json.dumps({"input_lens": [1, 2], "output_lens": [3, 4]})
            )
            match = from_paths(path, bench)
            self.assertEqual(match.timed_trace_count, 2)
            self.assertTrue(match.ok)

    def test_count_flag_prints_line_count(self) -> None:
        """--count prints nonempty jsonl lines and skips bench."""
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            _write_trace(trace, 3)
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(["--count", str(trace)])
            self.assertEqual(code, 0)
            self.assertEqual(buf.getvalue(), "3\n")


if __name__ == "__main__":
    unittest.main()
