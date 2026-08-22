#!/usr/bin/env python3
"""Local tests for generate_workload. Stdlib unittest; runs locally."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goodput_lab.generate_workload import (
    DECODE_SHAPE,
    TIMED_TRACE_KEYS,
    Phase,
    decode_profile_rows,
    main,
    write_jsonl,
)
from goodput_lab.count_match import count_nonempty_lines
from goodput_lab.timed_trace import CHUNK_HASH_SIZE, dummy_chunk_ids


class TestDecodeProfileRows(unittest.TestCase):
    """decode-envelope rows are deterministic and timed_trace-shaped."""

    def test_seed_42_is_deterministic_with_timed_trace_keys_and_phase(self) -> None:
        """Fixed seed repeats; vLLM keys plus phase; timestamps and phases in order."""
        rows = decode_profile_rows(42)
        again = decode_profile_rows(42)
        self.assertEqual(rows, again)
        other = decode_profile_rows(43)
        self.assertNotEqual(rows, other)
        self.assertEqual(len(rows), 465)

        timestamps = [row["timestamp"] for row in rows]
        self.assertEqual(timestamps[0], 0.0)
        self.assertEqual(timestamps, sorted(timestamps))

        phases = []
        for row in rows:
            self.assertEqual(set(row), set(TIMED_TRACE_KEYS) | {"phase"})
            self.assertIn(row["phase"], ("healthy", "busy", "pressure", "recovery"))
            self.assertIsInstance(row["timestamp"], float)
            self.assertIsInstance(row["input_length"], int)
            self.assertIsInstance(row["output_length"], int)
            self.assertIsInstance(row["hash_ids"], list)
            self.assertGreaterEqual(row["input_length"], DECODE_SHAPE.input_len_min)
            self.assertLessEqual(row["input_length"], DECODE_SHAPE.input_len_max)
            self.assertGreaterEqual(
                row["output_length"], DECODE_SHAPE.output_len_min
            )
            self.assertLessEqual(row["output_length"], DECODE_SHAPE.output_len_max)
            if not phases or phases[-1] != row["phase"]:
                phases.append(row["phase"])
        self.assertEqual(phases, ["healthy", "busy", "pressure", "recovery"])

        next_hash = 1
        for row in rows:
            expected = dummy_chunk_ids(
                row["input_length"], CHUNK_HASH_SIZE, next_hash
            )
            self.assertEqual(row["hash_ids"], expected)
            next_hash += len(expected)

    def test_count_match_count_equals_rows_written(self) -> None:
        """count_match nonempty line count equals the number of jsonl rows written."""
        rows = decode_profile_rows(42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            n_written = write_jsonl(path, rows)
            self.assertEqual(n_written, len(rows))
            self.assertEqual(count_nonempty_lines(path), n_written)
            loaded = [
                json.loads(line) for line in path.read_text().splitlines() if line
            ]
            self.assertEqual(len(loaded), n_written)
            self.assertIn("input_length", loaded[0])
            self.assertIn("output_length", loaded[0])
            self.assertIn("phase", loaded[0])

    def test_cli_writes_jsonl(self) -> None:
        """CLI writes jsonl and returns 0."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(
                    [
                        "--profile",
                        "decode-envelope",
                        "--seed",
                        "42",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            self.assertFalse(out.with_suffix(".meta.json").exists())
            self.assertEqual(count_nonempty_lines(out), len(decode_profile_rows(42)))
            stdout = buf.getvalue()
            self.assertIn("profile=decode-envelope", stdout)
            self.assertIn("seed=42", stdout)

    def test_per_phase_interarrival_and_length_band(self) -> None:
        """Timestamp deltas equal that phase's interarrival; lengths stay in the envelope."""
        rows = decode_profile_rows(42)
        self.assertEqual(DECODE_SHAPE.interarrival[Phase.HEALTHY], 2.0)
        self.assertEqual(DECODE_SHAPE.interarrival[Phase.BUSY], 0.5)
        self.assertEqual(DECODE_SHAPE.interarrival[Phase.PRESSURE], 0.08)
        self.assertEqual(DECODE_SHAPE.interarrival[Phase.RECOVERY], 2.0)
        by_phase: dict[str, list[dict]] = {}
        for row in rows:
            by_phase.setdefault(row["phase"], []).append(row)
        for phase in Phase:
            phase_rows = by_phase[phase.value]
            self.assertGreater(len(phase_rows), 1)
            gap = DECODE_SHAPE.interarrival[phase]
            for previous, current in zip(phase_rows, phase_rows[1:]):
                self.assertAlmostEqual(
                    current["timestamp"] - previous["timestamp"], gap
                )
            for row in phase_rows:
                self.assertGreaterEqual(
                    row["input_length"], DECODE_SHAPE.input_len_min
                )
                self.assertLessEqual(
                    row["input_length"], DECODE_SHAPE.input_len_max
                )
                self.assertGreaterEqual(
                    row["output_length"], DECODE_SHAPE.output_len_min
                )
                self.assertLessEqual(
                    row["output_length"], DECODE_SHAPE.output_len_max
                )


class TestDecodeShape(unittest.TestCase):
    """Decode-load length bands stay short-prompt, long-generate."""

    def test_input_len_max_below_output_len_min(self) -> None:
        """Decode load: input_len_max is below output_len_min in every phase's length band."""
        self.assertLess(DECODE_SHAPE.input_len_max, DECODE_SHAPE.output_len_min)


class TestProfileNotYet(unittest.TestCase):
    """Unimplemented profiles exit nonzero with not yet."""

    def test_prefill_blast_not_yet(self) -> None:
        """prefill-blast exits nonzero and stderr contains not yet."""
        self._assert_not_yet("prefill-blast")

    def test_ownership_not_yet(self) -> None:
        """ownership exits nonzero and stderr contains not yet."""
        self._assert_not_yet("ownership")

    def _assert_not_yet(self, profile: str) -> None:
        """Run CLI for profile and require a not-yet error without writing jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                code = main(
                    ["--profile", profile, "--seed", "1", "--out", str(out)]
                )
            self.assertEqual(code, 1)
            self.assertIn("not yet", buf.getvalue())
            self.assertIn(profile, buf.getvalue())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
