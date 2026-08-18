#!/usr/bin/env python3
"""Tiny unit tests for timed_trace helpers. Stdlib unittest."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import goodput_lab
from goodput_lab.request_window import (
    RecordedRequest,
    RequestWindow,
    first_arrival_time,
    from_csv,
    resolve_window_bounds,
    window_bounds,
)
from goodput_lab.window_policy import (
    PolicyInputs,
    PolicyReason,
    apply_knobs,
    build_profile,
    parse_args as parse_policy_args,
    profile_lines,
    recommend_policy,
    write_profile,
)
from goodput_lab.timed_trace import (
    TimedTraceRecord,
    build_timed_trace,
    dummy_chunk_ids,
    downsample,
    drop_over_max_model_len,
    parse_args as parse_trace_args,
    resolve_replay,
    to_timed_trace_records,
)

FIXTURE = Path(goodput_lab.__file__).resolve().parent / "fixtures" / "tiny_azure.csv"


class TestRecordedRequestFields(unittest.TestCase):
    def test_vllm_names_from_azure_columns(self) -> None:
        window = from_csv(FIXTURE)
        self.assertEqual(window.dropped_gen0, 1)
        self.assertEqual(len(window.rows), 9)
        r0 = window.rows[0]
        self.assertEqual(r0.input_length, 40)
        self.assertEqual(r0.output_length, 8)
        self.assertEqual(r0.total_tokens, 48)
        self.assertTrue(hasattr(r0, "arrival_time"))
        self.assertFalse(hasattr(r0, "ts"))
        self.assertFalse(hasattr(r0, "context"))
        self.assertFalse(hasattr(r0, "generated"))


def _policy_inputs(
    window_span: float,
    sample_size: int,
    arrival_scale: float,
    min_sample_size: int,
    max_runtime: float,
) -> PolicyInputs:
    """Build PolicyInputs so derived unscaled_rate / guessed_rate yield arrival_scale."""
    unscaled_rate = sample_size / window_span
    return PolicyInputs(
        window_span=window_span,
        sample_size=sample_size,
        guessed_rate=unscaled_rate / arrival_scale,
        min_sample_size=min_sample_size,
        max_runtime=max_runtime,
    )


class TestRequestWindowStats(unittest.TestCase):
    def test_unscaled_rate(self) -> None:
        window = from_csv(FIXTURE)
        span = (
            window.rows[-1].arrival_time - window.rows[0].arrival_time
        ).total_seconds()
        self.assertAlmostEqual(span, 1.8)
        self.assertAlmostEqual(len(window.rows) / span, 9 / 1.8)


class TestRecommendPolicy(unittest.TestCase):
    def test_need_longer_window(self) -> None:
        pol = recommend_policy(
            _policy_inputs(60.0, 100, 1.0, min_sample_size=1000, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.NEED_LONGER_RAW_WINDOW)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.arrival_scale, 1.0)
        self.assertEqual(pol.estimated_runtime, 60.0)

    def test_keep_all_fits(self) -> None:
        pol = recommend_policy(
            _policy_inputs(60.0, 2000, 1.0, min_sample_size=1000, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.KEEP_ALL_FITS)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.arrival_scale, 1.0)
        self.assertEqual(pol.estimated_runtime, 60.0)

    def test_scale_only(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 2000, 1.0, min_sample_size=1000, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.SCALE_ONLY)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.arrival_scale, 1.0)
        self.assertEqual(pol.estimated_runtime, 300.0)

    def test_stride(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 2000, 10.0, min_sample_size=100, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.DOWNSAMPLE)
        self.assertEqual(pol.stride, 3)
        self.assertEqual(pol.arrival_scale, 10.0)
        self.assertEqual(pol.estimated_runtime, 3000.0)
        self.assertEqual(pol.arrival_scale_after_downsample, 10.0 / 3)
        self.assertEqual(pol.estimated_runtime_after_downsample, 1000.0)
        self.assertEqual(pol.sample_size_after_downsample, 667)  # [::stride], not floor 2000//3=666

    def test_stride_backs_off_when_min_sample_size_binds(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 200, 10.0, min_sample_size=150, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.SHRINK_MIN_SAMPLE_SIZE)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.estimated_runtime, 3000.0)
        self.assertIsNone(pol.sample_size_after_downsample)

    def test_shrink_min_sample_size(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 100, 100.0, min_sample_size=80, max_runtime=1200.0)
        )
        self.assertEqual(pol.reason, PolicyReason.SHRINK_MIN_SAMPLE_SIZE)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.arrival_scale, 100.0)
        self.assertEqual(pol.estimated_runtime, 30000.0)
        self.assertIsNone(pol.sample_size_after_downsample)
        self.assertIsNone(pol.arrival_scale_after_downsample)

    def test_shrink_min_sample_size_reuses_stride_and_apply_knobs(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 1000, 10.0 / 0.6, min_sample_size=400, max_runtime=1000.0)
        )
        self.assertEqual(pol.reason, PolicyReason.SHRINK_MIN_SAMPLE_SIZE)
        self.assertEqual(pol.stride, 2)
        self.assertEqual(pol.sample_size_after_downsample, 500)
        self.assertEqual(pol.arrival_scale_after_downsample, (10.0 / 0.6) / 2)
        self.assertEqual(pol.estimated_runtime_after_downsample, 2500.0)

    def test_time_fit_without_guessed_rate_short_window(self) -> None:
        pol = recommend_policy(
            PolicyInputs(
                window_span=60.0,
                sample_size=2000,
                guessed_rate=None,
                min_sample_size=1000,
                max_runtime=1200.0,
            )
        )
        self.assertEqual(pol.reason, PolicyReason.KEEP_ALL_FITS)
        self.assertEqual(pol.arrival_scale, 1.0)
        self.assertEqual(pol.stride, 1)
        self.assertEqual(pol.estimated_runtime, 60.0)

    def test_time_fit_without_guessed_rate_fits_max_runtime(self) -> None:
        pol = recommend_policy(
            PolicyInputs(
                window_span=3600.0,
                sample_size=2000,
                guessed_rate=None,
                min_sample_size=100,
                max_runtime=1200.0,
            )
        )
        runtime = pol.estimated_runtime_after_downsample or pol.estimated_runtime
        self.assertLessEqual(runtime, 1200.0)
        self.assertEqual(pol.arrival_scale, 1.0)
        self.assertEqual(pol.stride, 3)
        self.assertEqual(pol.reason, PolicyReason.DOWNSAMPLE)
        self.assertEqual(pol.estimated_runtime, 3600.0)
        self.assertEqual(pol.estimated_runtime_after_downsample, 1200.0)


class TestReplayProfile(unittest.TestCase):
    def test_fixture_policy_without_cli_main(self) -> None:
        window = from_csv(FIXTURE)
        profile = build_profile(
            window,
            csv_path=FIXTURE,
            start=None,
            end=None,
            max_model_len=2048,
            min_sample_size=4,
            max_runtime=60.0,
            guessed_rate=2.0,
        )
        pol = profile["policy"]
        self.assertEqual(pol["reason"], PolicyReason.KEEP_ALL_FITS.value)
        self.assertEqual(pol["stride"], 1)
        self.assertAlmostEqual(pol["arrival_scale"], 8 / 1.8 / 2.0)
        self.assertAlmostEqual(pol["estimated_runtime"], 1.8 * (8 / 1.8 / 2.0))
        self.assertAlmostEqual(profile["apply"]["arrival_scale"], 8 / 1.8 / 2.0)
        self.assertEqual(profile["apply"]["stride"], 1)
        self.assertEqual(profile["apply"]["max_model_len"], 2048)
        self.assertEqual(profile["inputs"]["sample_size"], 8)
        self.assertEqual(profile["inputs"]["min_sample_size"], 4)
        self.assertEqual(profile["inputs"]["max_runtime"], 60.0)
        self.assertEqual(profile["inputs"]["guessed_rate"], 2.0)
        self.assertNotIn("sample_size", pol)
        self.assertNotIn("min_sample_size", pol)
        self.assertNotIn("max_runtime", pol)
        self.assertNotIn("guessed_rate", pol)
        self.assertNotIn("window_span", pol)
        self.assertNotIn("window_policy", pol)
        self.assertNotIn("timescale_kept", pol)
        self.assertNotIn("length_stats", profile)
        text = profile_lines(profile)
        self.assertIn("reason={}".format(PolicyReason.KEEP_ALL_FITS.value), text)
        self.assertIn("--arrival-scale=2.2222", text)
        self.assertIn("--stride=1", text)
        self.assertIn("estimated_runtime=4.0", text)
        self.assertIn("sample_size=8", text)

    def test_build_profile_without_guessed_rate_fits_max_runtime(self) -> None:
        window = from_csv(FIXTURE)
        profile = build_profile(
            window,
            csv_path=FIXTURE,
            start=None,
            end=None,
            max_model_len=2048,
            min_sample_size=4,
            max_runtime=60.0,
        )
        pol = profile["policy"]
        runtime = pol.get(
            "estimated_runtime_after_downsample", pol["estimated_runtime"]
        )
        self.assertLessEqual(runtime, 60.0)
        self.assertIsNone(profile["inputs"]["guessed_rate"])
        self.assertEqual(pol["arrival_scale"], 1.0)
        self.assertEqual(pol["stride"], 1)
        self.assertEqual(pol["reason"], PolicyReason.KEEP_ALL_FITS.value)

    def test_cli_guessed_rate_optional(self) -> None:
        args = parse_policy_args(["--out", "x.json"])
        self.assertIsNone(args.guessed_rate)
        self.assertEqual(args.max_runtime, 1200.0)

    def test_apply_block_uses_arrival_scale_over_stride(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 2000, 10.0, min_sample_size=100, max_runtime=1200.0)
        )
        apply = apply_knobs(pol)
        self.assertEqual(pol.stride, 3)
        self.assertAlmostEqual(apply["arrival_scale"], 10.0 / 3)
        self.assertEqual(apply["stride"], 3)
        profile = {
            "csv": "x",
            "start": None,
            "end": None,
            "inputs": {
                "guessed_rate": 1.0,
                "max_runtime": 1200.0,
                "min_sample_size": 100,
                "window_span": 300.0,
                "sample_size": 2000,
                "max_model_len": None,
            },
            "policy": pol.to_dict(),
            "apply": apply,
            "explanation": pol.reason,
        }
        text = profile_lines(profile)
        self.assertIn("--arrival-scale={:.4f}".format(10.0 / 3), text)
        self.assertIn("--stride=3", text)
        self.assertIn("sample_size=667", text)
        self.assertNotIn("--arrival-scale=10.0000", text)

    def test_timed_trace_profile_uses_apply_arrival_scale(self) -> None:
        pol = recommend_policy(
            _policy_inputs(300.0, 2000, 10.0, min_sample_size=100, max_runtime=1200.0)
        )
        apply = apply_knobs(pol)
        apply["max_model_len"] = 2048
        profile = {
            "csv": str(FIXTURE),
            "start": None,
            "end": None,
            "inputs": {
                "guessed_rate": 1.0,
                "max_runtime": 1200.0,
                "min_sample_size": 100,
                "window_span": 300.0,
                "sample_size": 2000,
                "max_model_len": 2048,
            },
            "policy": pol.to_dict(),
            "apply": apply,
            "explanation": pol.reason,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            out = Path(tmp) / "out.jsonl"
            write_profile(path, profile)
            args = resolve_replay(
                parse_trace_args(["--profile", str(path), "--out", str(out)])
            )
            self.assertAlmostEqual(args.arrival_scale, 10.0 / 3)
            self.assertEqual(args.stride, 3)
            self.assertEqual(args.max_model_len, 2048)
            self.assertEqual(Path(args.csv), FIXTURE)

            overridden = resolve_replay(
                parse_trace_args(
                    [
                        "--profile",
                        str(path),
                        "--arrival-scale",
                        "1.5",
                        "--out",
                        str(out),
                    ]
                )
            )
            self.assertEqual(overridden.arrival_scale, 1.5)
            self.assertEqual(overridden.stride, 3)


class TestDummyChunkIds(unittest.TestCase):
    def test_chunks(self) -> None:
        self.assertEqual(dummy_chunk_ids(0, 16, 1), [])
        self.assertEqual(dummy_chunk_ids(16, 16, 1), [1])
        self.assertEqual(dummy_chunk_ids(17, 16, 1), [1, 2])


class TestDownsample(unittest.TestCase):
    def test_stride(self) -> None:
        rows = list(range(10))
        self.assertEqual(downsample(rows, 1), list(range(10)))
        self.assertEqual(downsample(rows, 2), [0, 2, 4, 6, 8])
        self.assertEqual(downsample(rows, 3), [0, 3, 6, 9])


class TestDropOverMaxModelLen(unittest.TestCase):
    def test_fixture(self) -> None:
        window = from_csv(FIXTURE)
        kept, counts = drop_over_max_model_len(window.rows, 2048)
        self.assertEqual(counts["n_dropped_empty"], 0)
        self.assertEqual(counts["n_dropped_over_max_model_len"], 1)
        self.assertEqual(counts["n_after_fit"], 8)
        self.assertEqual(len(kept), 8)
        self.assertTrue(all(r.total_tokens <= 2048 for r in kept))


class TestBuildTimedTrace(unittest.TestCase):
    def test_always_stages(self) -> None:
        window = from_csv(FIXTURE)
        records = build_timed_trace(
            window.rows,
            max_model_len=2048,
            arrival_scale=10,
            stride=1,
        )
        self.assertEqual(len(records), 8)
        self.assertTrue(all(isinstance(r, TimedTraceRecord) for r in records))
        self.assertEqual(records[0].timestamp, 0.0)
        self.assertEqual(records[0].input_length, 40)
        self.assertEqual(records[0].output_length, 8)
        self.assertEqual(records[0].azure_generated, 8)
        self.assertTrue(all(r.output_length == r.azure_generated for r in records))
        self.assertTrue(all(r.input_length + r.azure_generated <= 2048 for r in records))

    def test_keep_long_prompt_tail_is_optional_other_file(self) -> None:
        window = from_csv(FIXTURE)
        records = build_timed_trace(
            window.rows,
            max_model_len=2048,
            arrival_scale=1,
            stride=1,
            keep_long_prompt_tail=True,
        )
        self.assertGreaterEqual(len(records), 1)
        self.assertLess(len(records), 8)
        self.assertTrue(all(r.output_length == 1 for r in records))
        self.assertTrue(all(r.azure_generated >= 1 for r in records))
        main = build_timed_trace(
            window.rows,
            max_model_len=2048,
            arrival_scale=1,
            stride=1,
        )
        hol_min_ctx = min(r.input_length for r in records)
        self.assertTrue(any(r.input_length < hol_min_ctx for r in main))

    def test_stride_matches_policy_sample_size_after_downsample(self) -> None:
        window = from_csv(FIXTURE)
        kept, _ = drop_over_max_model_len(window.rows, 2048)
        stride = 2
        sample_size_after_downsample = len(kept[::stride])
        records = build_timed_trace(
            window.rows,
            max_model_len=2048,
            arrival_scale=1.0,
            stride=stride,
        )
        self.assertEqual(len(records), sample_size_after_downsample)


class TestToTimedTraceRecords(unittest.TestCase):
    def test_arrival_scale_after_first_row(self) -> None:
        window = from_csv(FIXTURE)
        records = to_timed_trace_records(
            window.rows, arrival_scale=10.0, chunk_size=16
        )
        self.assertEqual(records[0].timestamp, 0.0)
        dt = (
            window.rows[1].arrival_time - window.rows[0].arrival_time
        ).total_seconds()
        self.assertAlmostEqual(records[1].timestamp, dt * 10.0)
        self.assertAlmostEqual(records[1].timestamp, 2.0)


class TestFromCsvWindow(unittest.TestCase):
    def test_start_inclusive_end_exclusive(self) -> None:
        start = datetime(2024, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 5, 10, 0, 0, 0, 200000, tzinfo=timezone.utc)
        window = from_csv(FIXTURE, start=start, end=end)
        self.assertEqual(len(window.rows), 1)
        self.assertEqual(window.rows[0].input_length, 40)

        end2 = datetime(2024, 5, 10, 0, 0, 0, 400000, tzinfo=timezone.utc)
        window = from_csv(FIXTURE, start=start, end=end2)
        self.assertEqual(len(window.rows), 2)
        self.assertEqual(window.rows[1].input_length, 80)

    def test_window_sec_without_start_needs_default_start(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            window_bounds(None, None, 60.0)
        self.assertIn("first TIMESTAMP", str(ctx.exception))

    def test_window_sec_without_start_uses_default_start(self) -> None:
        first = datetime(2024, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
        t0, t1 = window_bounds(None, None, 60.0, default_start=first)
        self.assertEqual(t0, first)
        self.assertEqual((t1 - t0).total_seconds(), 60.0)

    def test_end_and_window_sec_conflict(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            window_bounds(
                "2024-05-10T00:00:00+00:00",
                "2024-05-10T00:01:00+00:00",
                60.0,
            )
        self.assertIn("pass only one of --end and --window-sec", str(ctx.exception))

    def test_start_plus_window_sec(self) -> None:
        t0, t1 = window_bounds("2024-05-10T00:00:00+00:00", None, 60.0)
        self.assertIsNotNone(t0)
        self.assertIsNotNone(t1)
        self.assertEqual((t1 - t0).total_seconds(), 60.0)

    def test_cli_window_sec_without_start_ok(self) -> None:
        args = parse_policy_args(
            [
                "--window-sec",
                "60",
                "--out",
                "x.json",
            ]
        )
        self.assertEqual(args.window_sec, 60.0)
        self.assertIsNone(args.start)
        self.assertIsNone(args.guessed_rate)

    def test_window_sec_alone_is_first_n_seconds_of_csv(self) -> None:
        first = first_arrival_time(FIXTURE)
        self.assertEqual(
            first, datetime(2024, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
        )
        start, end = resolve_window_bounds(FIXTURE, None, None, 1.0)
        self.assertEqual(start, first)
        self.assertEqual((end - start).total_seconds(), 1.0)
        window = from_csv(FIXTURE, start, end)
        self.assertEqual(len(window.rows), 4)


class TestFilterAlignment(unittest.TestCase):
    def test_empty_prompt_dropped_like_timed_trace(self) -> None:
        t0 = datetime(2024, 5, 10, tzinfo=timezone.utc)
        rows = [
            RecordedRequest(0, t0, 0, 8),
            RecordedRequest(1, t0 + timedelta(seconds=1), 10, 8),
            RecordedRequest(2, t0 + timedelta(seconds=2), 100, 50),
        ]
        max_len = 50
        kept, counts = drop_over_max_model_len(rows, max_len)
        self.assertEqual(counts["n_dropped_empty"], 1)
        self.assertEqual(counts["n_dropped_over_max_model_len"], 1)
        self.assertEqual(len(kept), 1)
        window = RequestWindow(rows=rows, scanned=3, dropped_gen0=0)
        profile = build_profile(
            window,
            csv_path=FIXTURE,
            start=None,
            end=None,
            max_model_len=max_len,
            min_sample_size=1,
            max_runtime=60.0,
            guessed_rate=1.0,
        )
        self.assertEqual(profile["inputs"]["sample_size"], 1)
        records = build_timed_trace(
            rows,
            max_model_len=max_len,
            arrival_scale=1.0,
            stride=1,
        )
        self.assertEqual(len(records), profile["inputs"]["sample_size"])


if __name__ == "__main__":
    unittest.main()
