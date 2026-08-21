#!/usr/bin/env python3
"""Local tests for results join and CLI. Stdlib unittest; runs locally."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goodput_lab.bench_result import BenchResult
from goodput_lab.config import DEFAULT_PIN_PATH
from goodput_lab.metrics_jsonl import (
    KV_SERIES,
    PREEMPTION_SERIES,
    RUNNING_SERIES,
    Scrape,
)
from goodput_lab.results import (
    main,
    _results_from_rows,
    results_from_run_dir,
    rows_from_timed_trace,
    write_results,
)
from goodput_lab.serve_settings import from_run_dir as serve_settings_from_run_dir
from goodput_lab.serve_settings import write as write_serve_settings


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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def _trace_row(timestamp: float, index: int, phase: str) -> dict:
    """One generated-shaped row with a phase key."""
    return {
        "timestamp": timestamp,
        "input_length": 16,
        "output_length": 8,
        "hash_ids": [index + 1],
        "phase": phase,
    }


def _scrape(
    unix_ms: int, running_n: float, kv: float = 0.0, preemptions: float = 0.0
) -> Scrape:
    """One scrape with pin series values."""
    return Scrape(
        unix_ms=unix_ms,
        gauges={RUNNING_SERIES: running_n, KV_SERIES: kv},
        counters={PREEMPTION_SERIES: preemptions},
    )


def _bench(
    n: int,
    ttfts: list | None = None,
    itls: list | None = None,
    duration: float | None = 10.0,
    start_times: list | None = None,
) -> BenchResult:
    """n-request bench aligned to a tiny trace. duration 10s matches last-running math."""
    return BenchResult(
        input_lens=[16] * n,
        output_lens=[8] * n,
        ttfts=ttfts if ttfts is not None else [0.010] * n,
        itls=itls if itls is not None else [[0.002]] * n,
        queue_times=None,
        start_times=start_times,
        duration=duration,
    )


def _write_fixture(root: Path) -> tuple[Path, Path]:
    """Fake run dir plus two-phase jsonl. Returns (run_dir, trace)."""
    run_dir = root / "run"
    run_dir.mkdir()
    trace = root / "trace.jsonl"
    rows = [
        _trace_row(0.0, 0, "healthy"),
        _trace_row(1.0, 1, "healthy"),
        _trace_row(10.0, 2, "busy"),
    ]
    _write_jsonl(trace, rows)
    (run_dir / "bench.json").write_text(
        json.dumps(
            {
                "input_lens": [16, 16, 16],
                "output_lens": [8, 8, 8],
                "ttfts": [0.010, 0.010, 0.100],
                "itls": [[0.002, 0.002], [0.002], [0.004]],
                "duration": 10.0,
            }
        )
    )
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"timestamp": 1_000_000, "body": _prom_body(0.0, 0.0, 2.0)},
            {"timestamp": 1_001_000, "body": _prom_body(1.0, 0.1, 2.0)},
            {"timestamp": 1_002_000, "body": _prom_body(3.0, 0.2, 5.0)},
            {"timestamp": 1_011_000, "body": _prom_body(8.0, 0.5, 5.0)},
        ],
    )
    (run_dir / "pin.yaml").write_text(DEFAULT_PIN_PATH.read_text())
    (run_dir / "serve.cmd").write_text(
        "serve: vllm serve Qwen/Qwen2.5-7B-Instruct "
        "--host 127.0.0.1 --port 8000 --dtype bfloat16 "
        "--no-enable-prefix-caching --max-model-len 32768\n"
    )
    (run_dir / "serve.log").write_text(
        "INFO Engine arguments: EngineArgs(model='Qwen/Qwen2.5-7B-Instruct', "
        "max_num_seqs=256, dtype='bfloat16')\n"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"git_head": "abc123", "chunk_hash_size": 16, "mode": "discover"},
            indent=2,
        )
        + "\n"
    )
    return run_dir, trace


class TestResultsFromRunDir(unittest.TestCase):
    """Orchestrator loads run dir artifacts and returns Results."""

    def test_healthy_percentiles_from_run_dir(self) -> None:
        """results_from_run_dir joins bench to phases and windows scrapes from first send."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            results = results_from_run_dir(run_dir, trace)
            self.assertEqual(results.healthy.ttft_p50, 10.0)
            self.assertEqual(results.healthy.window_stats.preemption_delta, 3.0)
            self.assertIsNotNone(results.busy)
            self.assertEqual(results.busy.ttft_p50, 100.0)


class TestCli(unittest.TestCase):
    """CLI writes both JSON artifacts into the run dir."""

    def test_cli_writes_both_json_files(self) -> None:
        """CLI writes results.json and serve-settings.json."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main(["--run-dir", str(run_dir), "--trace", str(trace)])
            self.assertEqual(code, 0)
            results_path = run_dir / "results.json"
            settings_path = run_dir / "serve-settings.json"
            self.assertTrue(results_path.is_file())
            self.assertTrue(settings_path.is_file())
            self.assertFalse((run_dir / "baseline.json").exists())
            payload = json.loads(results_path.read_text())
            self.assertNotIn("first_send_unix_ms", payload)
            self.assertNotIn("running_series", payload)
            self.assertNotIn("kv_series", payload)
            self.assertNotIn("preemption_series", payload)
            self.assertNotIn("t0_rule", payload)
            self.assertNotIn("t0_unix_ms", payload)
            self.assertNotIn("ttft_p50", payload)
            self.assertEqual(payload["healthy"]["ttft_p50"], 10.0)
            self.assertEqual(payload["healthy"]["preemption_delta"], 3.0)
            self.assertNotIn("queue_time_p50", payload["healthy"])

    def test_cli_writes_results_when_serve_settings_fails(self) -> None:
        """results.json is written first; serve-settings failure exits nonzero and names that artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            (run_dir / "metadata.json").unlink()
            err = io.StringIO()
            with patch("sys.stderr", err):
                code = main(["--run-dir", str(run_dir), "--trace", str(trace)])
            self.assertEqual(code, 1)
            self.assertTrue((run_dir / "results.json").is_file())
            self.assertFalse((run_dir / "serve-settings.json").exists())
            self.assertIn("serve-settings.json", err.getvalue())

    def test_cli_empty_scrape_window_errors(self) -> None:
        """CLI exits nonzero when a phase has rows and an empty scrape window."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            _write_jsonl(trace, [_trace_row(0.0, 0, "healthy")])
            (run_dir / "bench.json").write_text(
                json.dumps(
                    {
                        "input_lens": [16],
                        "output_lens": [8],
                        "ttfts": [0.010],
                        "itls": [[0.002]],
                        "duration": 10.0,
                    }
                )
            )
            _write_jsonl(
                run_dir / "metrics.jsonl",
                [
                    {"timestamp": 1_000_000, "body": _prom_body(1.0, 0.1, 0.0)},
                    {"timestamp": 1_020_000, "body": _prom_body(1.0, 0.1, 0.0)},
                ],
            )
            err = io.StringIO()
            with patch("sys.stderr", err):
                code = main(["--run-dir", str(run_dir), "--trace", str(trace)])
            self.assertEqual(code, 1)
            self.assertIn("results.json", err.getvalue())
            self.assertIn("scrape", err.getvalue())
            self.assertFalse((run_dir / "results.json").exists())


class TestWriteResults(unittest.TestCase):
    """write_results dumps results.json with indent."""

    def test_write_results_round_trip(self) -> None:
        """Written JSON matches Results.to_dict."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            results = results_from_run_dir(run_dir, trace)
            path = write_results(run_dir, results)
            self.assertEqual(json.loads(path.read_text()), results.to_dict())
            settings = serve_settings_from_run_dir(run_dir)
            settings_path = write_serve_settings(run_dir, settings)
            self.assertEqual(settings_path.name, "serve-settings.json")
            self.assertEqual(json.loads(settings_path.read_text()), settings.to_dict())


class TestResultsFromRows(unittest.TestCase):
    """Join ttfts/itls by phase; nest healthy stats on Results."""

    def test_healthy_percentiles_and_nested_busy(self) -> None:
        """Healthy TTFT/ITL nested under healthy; busy nested."""
        rows = [
            _trace_row(0.0, 0, "healthy"),
            _trace_row(1.0, 1, "healthy"),
            _trace_row(10.0, 2, "busy"),
        ]
        bench = BenchResult(
            input_lens=[16, 16, 16],
            output_lens=[8, 8, 8],
            ttfts=[0.010, 0.010, 0.100],
            itls=[[0.002, 0.002], [0.002], [0.004]],
            queue_times=None,
            duration=10.0,
        )
        scrapes = [
            _scrape(1_000_000, 0.0, 0.0, 2.0),
            _scrape(1_001_000, 1.0, 0.1, 2.0),
            _scrape(1_002_000, 3.0, 0.2, 5.0),
            _scrape(1_011_000, 8.0, 0.5, 5.0),
        ]
        results = _results_from_rows(rows, bench, scrapes)
        self.assertEqual(results.healthy.ttft_p50, 10.0)
        self.assertEqual(results.healthy.itl_p50, 2.0)
        self.assertEqual(results.healthy.window_stats.preemption_delta, 3.0)
        self.assertEqual(results.healthy.window_stats.running_median, 2.0)
        self.assertAlmostEqual(results.healthy.window_stats.kv_median, 0.15)
        self.assertIsNone(results.healthy.queue_time_p50)
        self.assertIsNotNone(results.busy)
        self.assertEqual(results.busy.ttft_p50, 100.0)
        self.assertEqual(results.busy.window_stats.preemption_delta, 0.0)
        self.assertIsNone(results.pressure)
        self.assertIsNone(results.recovery)

        payload = results.to_dict()
        self.assertNotIn("first_send_unix_ms", payload)
        self.assertNotIn("running_series", payload)
        self.assertNotIn("kv_series", payload)
        self.assertNotIn("preemption_series", payload)
        self.assertNotIn("t0_rule", payload)
        self.assertNotIn("t0_unix_ms", payload)
        self.assertNotIn("ttft_p50", payload)
        self.assertEqual(payload["healthy"]["ttft_p50"], 10.0)
        self.assertEqual(payload["healthy"]["running_median"], 2.0)
        self.assertNotIn("queue_time_p50", payload["healthy"])
        self.assertEqual(payload["busy"]["ttft_p50"], 100.0)
        self.assertNotIn("pressure", payload)
        self.assertNotIn("recovery", payload)

    def test_first_send_skips_warmup_scrapes(self) -> None:
        """Last scrape with running > 0 minus duration is first send; earlier warmup scrapes are excluded."""
        rows = [_trace_row(0.0, 0, "healthy"), _trace_row(1.0, 1, "healthy")]
        bench = _bench(2, duration=10.0)
        scrapes = [
            _scrape(1_000_000, 3.0, 0.4, 1.0),
            _scrape(1_001_000, 3.0, 0.4, 1.0),
            _scrape(1_002_000, 0.0, 0.0, 1.0),
            _scrape(1_010_000, 1.0, 0.1, 1.0),
            _scrape(1_011_000, 2.0, 0.2, 2.0),
            _scrape(1_020_000, 1.0, 0.1, 2.0),
        ]
        results = _results_from_rows(rows, bench, scrapes)
        self.assertEqual(results.healthy.window_stats.running_median, 1.5)

    def test_missing_duration_errors(self) -> None:
        """Missing bench duration raises ValueError."""
        rows = [_trace_row(0.0, 0, "healthy")]
        bench = _bench(1, duration=None, start_times=None)
        scrapes = [
            _scrape(1_000_000, 2.0),
            _scrape(1_001_000, 0.0),
            _scrape(1_005_000, 1.0, 0.1, 0.0),
        ]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, scrapes)
        self.assertIn("duration", str(ctx.exception))

    def test_first_send_from_last_running_minus_duration(self) -> None:
        """Last scrape with running > 0 minus duration windows the first jsonl send."""
        rows = [_trace_row(0.0, 0, "healthy")]
        first_send_ms = 1_700_000_000_000
        bench = _bench(1, duration=10.0)
        scrapes = [
            _scrape(first_send_ms - 1_000, 4.0),
            _scrape(first_send_ms, 1.0, 0.1, 0.0),
            _scrape(first_send_ms + 10_000, 1.0, 0.1, 0.0),
        ]
        results = _results_from_rows(rows, bench, scrapes)
        self.assertEqual(results.healthy.window_stats.running_median, 1.0)

    def test_last_phase_window_excludes_cleanup(self) -> None:
        """Last phase ends when the last request finishes."""
        first_send_ms = 1_700_000_000_000
        rows = [
            _trace_row(0.0, 0, "healthy"),
            _trace_row(10.0, 1, "busy"),
        ]
        bench = BenchResult(
            input_lens=[16, 16],
            output_lens=[8, 8],
            ttfts=[0.010, 0.100],
            itls=[[0.002], [0.004]],
            queue_times=None,
            duration=30.0,
        )
        scrapes = [
            _scrape(first_send_ms, 1.0, 0.1, 2.0),
            _scrape(first_send_ms + 10_000, 8.0, 0.5, 5.0),
            _scrape(first_send_ms + 30_000, 50.0, 0.9, 99.0),
        ]
        results = _results_from_rows(rows, bench, scrapes)
        self.assertIsNotNone(results.busy)
        self.assertEqual(results.busy.window_stats.running_median, 8.0)
        self.assertEqual(results.busy.window_stats.preemption_delta, 0.0)
        self.assertNotEqual(results.busy.window_stats.running_median, 50.0)

    def test_last_phase_window_ignores_perf_counter_start_times(self) -> None:
        """Last phase uses jsonl timestamps; start_times are time.perf_counter() seconds.

        A cleanup scrape after the last request finishes stays outside the window.
        """
        first_send_ms = 1_700_000_000_000
        rows = [
            _trace_row(0.0, 0, "healthy"),
            _trace_row(10.0, 1, "busy"),
        ]
        bench = BenchResult(
            input_lens=[16, 16],
            output_lens=[8, 8],
            ttfts=[0.010, 0.100],
            itls=[[0.002], [0.004]],
            queue_times=None,
            start_times=[3847.123, 3877.123],
            duration=30.0,
        )
        scrapes = [
            _scrape(first_send_ms, 1.0, 0.1, 2.0),
            _scrape(first_send_ms + 10_000, 8.0, 0.5, 5.0),
            _scrape(first_send_ms + 30_000, 50.0, 0.9, 99.0),
        ]
        results = _results_from_rows(rows, bench, scrapes)
        self.assertIsNotNone(results.busy)
        self.assertEqual(results.busy.window_stats.running_median, 8.0)
        self.assertEqual(results.busy.window_stats.preemption_delta, 0.0)
        self.assertNotEqual(results.busy.window_stats.running_median, 50.0)

    def test_empty_scrape_window_errors(self) -> None:
        """A phase with rows and an empty scrape window raises ValueError."""
        rows = [_trace_row(0.0, 0, "healthy")]
        bench = _bench(1, duration=10.0)
        scrapes = [
            _scrape(1_000_000, 1.0),
            _scrape(1_020_000, 1.0),
        ]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, scrapes)
        self.assertIn("scrape", str(ctx.exception))

    def test_null_ttft_at_index_errors(self) -> None:
        """A None TTFT on a selected row raises ValueError."""
        rows = [_trace_row(0.0, 0, "healthy")]
        bench = _bench(1, ttfts=[None], duration=1.0)
        scrapes = [_scrape(1_000_000, 1.0), _scrape(1_001_000, 1.0)]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, scrapes)
        self.assertIn("ttft", str(ctx.exception))

    def test_null_itl_at_index_errors(self) -> None:
        """A None ITL on a selected row raises ValueError."""
        rows = [_trace_row(0.0, 0, "healthy")]
        bench = _bench(1, itls=[None], duration=1.0)
        scrapes = [_scrape(1_000_000, 1.0), _scrape(1_001_000, 1.0)]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, scrapes)
        self.assertIn("itl", str(ctx.exception))

    def test_repeat_copies_are_refused(self) -> None:
        """rep_*.json in the run dir raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, trace = _write_fixture(Path(tmp))
            (run_dir / "rep_1.json").write_text((run_dir / "bench.json").read_text())
            with self.assertRaises(ValueError) as ctx:
                results_from_run_dir(run_dir, trace)
            message = str(ctx.exception)
            self.assertIn("rep_1.json", message)
            self.assertIn("summarize one bench.json run", message)
            self.assertIn("omit repeat copies", message)

    def test_input_lens_count_mismatch(self) -> None:
        """Mismatched row count vs input_lens raises ValueError."""
        rows = [_trace_row(0.0, 0, "healthy"), _trace_row(1.0, 1, "healthy")]
        bench = _bench(1)
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, [_scrape(1_000_000, 1.0)])
        self.assertIn("input_lens", str(ctx.exception))

    def test_missing_ttfts_length(self) -> None:
        """ttfts shorter than the trace is an error."""
        rows = [_trace_row(0.0, 0, "healthy")]
        bench = BenchResult(
            input_lens=[16],
            output_lens=[8],
            ttfts=[],
            itls=[[0.002]],
            queue_times=None,
            duration=1.0,
        )
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, bench, [_scrape(1_000_000, 1.0)])
        self.assertIn("ttfts", str(ctx.exception))

    def test_no_healthy_rows(self) -> None:
        """A trace with only busy rows raises ValueError."""
        rows = [_trace_row(0.0, 0, "busy")]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(
                rows,
                _bench(1),
                [_scrape(1_000_000, 1.0), _scrape(1_010_000, 1.0)],
            )
        self.assertIn("healthy", str(ctx.exception))

    def test_empty_trace(self) -> None:
        """Empty jsonl is an error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("\n")
            with self.assertRaises(ValueError) as ctx:
                rows_from_timed_trace(path)
            self.assertIn("empty trace", str(ctx.exception))

    def test_running_never_positive_errors(self) -> None:
        """All-zero running scrapes raise ValueError."""
        rows = [_trace_row(0.0, 0, "healthy")]
        scrapes = [_scrape(1_000_000, 0.0)]
        with self.assertRaises(ValueError) as ctx:
            _results_from_rows(rows, _bench(1, duration=10.0), scrapes)
        self.assertIn(RUNNING_SERIES, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
