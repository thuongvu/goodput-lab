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
    PREFILL_SHAPE,
    TIMED_TRACE_KEYS,
    Phase,
    PrefillRole,
    decode_profile_rows,
    main,
    prefill_blast_rows,
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


class TestPrefillBlastRows(unittest.TestCase):
    """prefill-blast rows are deterministic, role-tagged, and timed_trace-shaped."""

    def test_seed_42_stable_row_count_and_roles(self) -> None:
        """Fixed seed repeats; vLLM keys plus phase; decode_stream, long_prefill, and arrival_probe present."""
        rows = prefill_blast_rows(42)
        again = prefill_blast_rows(42)
        self.assertEqual(rows, again)
        other = prefill_blast_rows(43)
        self.assertNotEqual(rows, other)
        self.assertEqual(len(rows), 47)

        timestamps = [row["timestamp"] for row in rows]
        self.assertEqual(timestamps[0], 0.0)
        self.assertEqual(timestamps, sorted(timestamps))

        phases = {row["phase"] for row in rows}
        self.assertEqual(
            phases, {"decode_stream", "long_prefill", "arrival_probe"}
        )
        by_phase: dict[str, list[dict]] = {}
        for row in rows:
            self.assertEqual(set(row), set(TIMED_TRACE_KEYS) | {"phase"})
            self.assertIn(
                row["phase"],
                ("decode_stream", "long_prefill", "arrival_probe"),
            )
            self.assertIsInstance(row["timestamp"], float)
            self.assertIsInstance(row["input_length"], int)
            self.assertIsInstance(row["output_length"], int)
            self.assertIsInstance(row["hash_ids"], list)
            by_phase.setdefault(row["phase"], []).append(row)
        self.assertEqual(
            len(by_phase["decode_stream"]), PREFILL_SHAPE.decode_stream_count
        )
        self.assertEqual(
            len(by_phase["long_prefill"]), PREFILL_SHAPE.long_prefill_count
        )
        self.assertEqual(
            len(by_phase["arrival_probe"]), PREFILL_SHAPE.arrival_probe_count
        )

        next_hash = 1
        for row in rows:
            expected = dummy_chunk_ids(
                row["input_length"], CHUNK_HASH_SIZE, next_hash
            )
            self.assertEqual(row["hash_ids"], expected)
            next_hash += len(expected)

    def test_token_bands_per_role(self) -> None:
        """decode_stream, long_prefill, and arrival_probe lengths stay in their calibration bands."""
        rows = prefill_blast_rows(42)
        for row in rows:
            phase = row["phase"]
            if phase == PrefillRole.DECODE_STREAM.value:
                self.assertGreaterEqual(
                    row["input_length"],
                    PREFILL_SHAPE.decode_stream_input_len_min,
                )
                self.assertLessEqual(
                    row["input_length"],
                    PREFILL_SHAPE.decode_stream_input_len_max,
                )
                self.assertGreaterEqual(
                    row["output_length"],
                    PREFILL_SHAPE.decode_stream_output_len_min,
                )
                self.assertLessEqual(
                    row["output_length"],
                    PREFILL_SHAPE.decode_stream_output_len_max,
                )
            elif phase == PrefillRole.LONG_PREFILL.value:
                self.assertGreaterEqual(
                    row["input_length"],
                    PREFILL_SHAPE.long_prefill_input_len_min,
                )
                self.assertLessEqual(
                    row["input_length"],
                    PREFILL_SHAPE.long_prefill_input_len_max,
                )
                self.assertEqual(
                    row["output_length"], PREFILL_SHAPE.long_prefill_output_len
                )
            else:
                self.assertEqual(phase, PrefillRole.ARRIVAL_PROBE.value)
                self.assertGreaterEqual(
                    row["input_length"],
                    PREFILL_SHAPE.arrival_probe_input_len_min,
                )
                self.assertLessEqual(
                    row["input_length"],
                    PREFILL_SHAPE.arrival_probe_input_len_max,
                )
                self.assertGreaterEqual(
                    row["output_length"],
                    PREFILL_SHAPE.arrival_probe_output_len_min,
                )
                self.assertLessEqual(
                    row["output_length"],
                    PREFILL_SHAPE.arrival_probe_output_len_max,
                )

    def test_long_prefill_while_decode_stream_in_flight(self) -> None:
        """decode_stream sends first; long_prefills follow; probes pack onto that window."""
        rows = prefill_blast_rows(42)
        decode_stream_times = [
            row["timestamp"]
            for row in rows
            if row["phase"] == "decode_stream"
        ]
        long_prefill_times = [
            row["timestamp"]
            for row in rows
            if row["phase"] == "long_prefill"
        ]
        arrival_probe_times = [
            row["timestamp"]
            for row in rows
            if row["phase"] == "arrival_probe"
        ]
        self.assertTrue(decode_stream_times)
        self.assertTrue(long_prefill_times)
        self.assertTrue(arrival_probe_times)
        self.assertLess(max(decode_stream_times), min(long_prefill_times))
        self.assertGreaterEqual(len(long_prefill_times), 2)
        last_decode_stream = max(decode_stream_times)
        delay = PREFILL_SHAPE.long_prefill_delay
        gap = PREFILL_SHAPE.long_prefill_interarrival
        for index, send in enumerate(long_prefill_times):
            self.assertAlmostEqual(
                send, last_decode_stream + delay + index * gap
            )
        self.assertEqual(long_prefill_times, [3.25, 5.25, 7.25])
        self.assertAlmostEqual(arrival_probe_times[0], 3.25)
        self.assertAlmostEqual(arrival_probe_times[-1], 7.90)
        probe_gap = PREFILL_SHAPE.arrival_probe_interarrival
        self.assertEqual(probe_gap, 0.15)
        for previous, current in zip(
            arrival_probe_times, arrival_probe_times[1:]
        ):
            self.assertAlmostEqual(current - previous, probe_gap)
        for send in long_prefill_times:
            self.assertLessEqual(min(arrival_probe_times), send)
            self.assertGreater(max(arrival_probe_times), send)

    def test_t0_min_decode_stream_outlives_last_long_prefill(self) -> None:
        """Last long_prefill send is before a t=0 min-output decode_stream finishes.

        Assumes ~12 ms/token decode. Generate length versus send time.
        """
        rows = prefill_blast_rows(42)
        last_prefill = max(
            row["timestamp"]
            for row in rows
            if row["phase"] == "long_prefill"
        )
        decode_duration = PREFILL_SHAPE.decode_stream_output_len_min * 0.012
        self.assertLess(last_prefill, decode_duration)

    def test_cli_writes_jsonl(self) -> None:
        """CLI writes prefill-blast jsonl and returns 0."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(
                    [
                        "--profile",
                        "prefill-blast",
                        "--seed",
                        "42",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            self.assertFalse(out.with_suffix(".meta.json").exists())
            self.assertEqual(count_nonempty_lines(out), len(prefill_blast_rows(42)))
            stdout = buf.getvalue()
            self.assertIn("profile=prefill-blast", stdout)
            self.assertIn("seed=42", stdout)
            loaded = [
                json.loads(line) for line in out.read_text().splitlines() if line
            ]
            self.assertEqual(
                {row["phase"] for row in loaded},
                {"decode_stream", "long_prefill", "arrival_probe"},
            )


class TestPrefillShape(unittest.TestCase):
    """Prefill-blast length bands stay short decode_stream, fat long_prefill, tiny arrival_probe."""

    def test_probe_overlap_schedule(self) -> None:
        """Probes start at the first long_prefill send; 0.15 s between arrival probes."""
        self.assertEqual(PREFILL_SHAPE.decode_stream_count, 12)
        self.assertEqual(PREFILL_SHAPE.decode_stream_interarrival, 0.25)
        self.assertEqual(PREFILL_SHAPE.long_prefill_count, 3)
        self.assertEqual(PREFILL_SHAPE.long_prefill_delay, 0.5)
        self.assertEqual(PREFILL_SHAPE.long_prefill_interarrival, 2.0)
        self.assertEqual(PREFILL_SHAPE.arrival_probe_count, 32)
        self.assertEqual(PREFILL_SHAPE.first_arrival_probe, 3.25)
        self.assertEqual(PREFILL_SHAPE.arrival_probe_interarrival, 0.15)

    def test_role_bands(self) -> None:
        """decode_stream prompt is short vs its output; long_prefill prompt is large vs output 1; arrival_probe is tiny."""
        self.assertLess(
            PREFILL_SHAPE.decode_stream_input_len_max,
            PREFILL_SHAPE.decode_stream_output_len_min,
        )
        self.assertGreater(
            PREFILL_SHAPE.long_prefill_input_len_min,
            PREFILL_SHAPE.decode_stream_input_len_max,
        )
        self.assertEqual(PREFILL_SHAPE.long_prefill_output_len, 1)
        self.assertLess(
            PREFILL_SHAPE.arrival_probe_input_len_max,
            PREFILL_SHAPE.decode_stream_input_len_min,
        )
        self.assertLess(
            PREFILL_SHAPE.arrival_probe_output_len_max,
            PREFILL_SHAPE.decode_stream_output_len_min,
        )

    def test_prompt_plus_output_fits_max_model_len(self) -> None:
        """Prompt plus output fits pin.yaml max_model_len 32768."""
        rows = prefill_blast_rows(42)
        for row in rows:
            self.assertLessEqual(
                row["input_length"] + row["output_length"], 32768
            )


class TestProfileNotYet(unittest.TestCase):
    """Unimplemented profiles exit nonzero with not yet."""

    def test_ownership_not_yet(self) -> None:
        """ownership exits nonzero and stderr contains not yet."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                code = main(
                    ["--profile", "ownership", "--seed", "1", "--out", str(out)]
                )
            self.assertEqual(code, 1)
            self.assertIn("not yet", buf.getvalue())
            self.assertIn("ownership", buf.getvalue())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
