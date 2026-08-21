#!/usr/bin/env python3
"""Local tests for serve_settings. Stdlib unittest; runs locally."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from goodput_lab.config import DEFAULT_PIN_PATH
from goodput_lab.serve_settings import from_run_dir, write


def _write_run_dir(root: Path) -> Path:
    """Minimal run dir with pin, serve.cmd, serve.log, and metadata.json."""
    run_dir = root / "run"
    run_dir.mkdir()
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
    return run_dir


class TestFromRunDir(unittest.TestCase):
    """from_run_dir reads pin, serve.cmd, metadata, and serve.log."""

    def test_pin_serve_and_stock_defaults(self) -> None:
        """max_num_seqs comes from serve.log; missing keys are vllm_default_not_passed."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_run_dir(Path(tmp))
            settings = from_run_dir(run_dir)
            self.assertEqual(settings.model.name, "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(settings.model.dtype, "bfloat16")
            self.assertEqual(settings.model.max_model_len, 32768)
            self.assertEqual(settings.serve.host, "127.0.0.1")
            self.assertEqual(settings.serve.port, 8000)
            self.assertTrue(settings.serve.no_enable_prefix_caching)
            self.assertEqual(settings.metadata.chunk_hash_size, 16)
            self.assertEqual(settings.metadata.mode, "discover")
            self.assertEqual(settings.max_num_seqs.value, 256)
            self.assertEqual(settings.max_num_seqs.source, "serve.log")
            self.assertIsNone(settings.max_num_batched_tokens.value)
            self.assertEqual(
                settings.max_num_batched_tokens.source, "vllm_default_not_passed"
            )
            payload = settings.to_dict()
            self.assertEqual(
                payload["max_num_batched_tokens"],
                {"source": "vllm_default_not_passed"},
            )
            self.assertNotIn("value", payload["max_num_batched_tokens"])


class TestWrite(unittest.TestCase):
    """write dumps serve-settings.json."""

    def test_write_round_trip(self) -> None:
        """Written JSON matches to_dict."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_run_dir(Path(tmp))
            settings = from_run_dir(run_dir)
            path = write(run_dir, settings)
            self.assertEqual(path.name, "serve-settings.json")
            self.assertEqual(json.loads(path.read_text()), settings.to_dict())


if __name__ == "__main__":
    unittest.main()
