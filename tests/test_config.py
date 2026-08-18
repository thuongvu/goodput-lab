#!/usr/bin/env python3
"""Smoke tests for config/pin.yaml load + shell export flattening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from goodput_lab.config import (
    DEFAULT_PIN_PATH,
    load_pin,
    shell_exports,
    write_max_model_len,
)


class TestLoadPin(unittest.TestCase):
    def test_yaml_loads_nested_sections(self) -> None:
        data = load_pin(DEFAULT_PIN_PATH)
        self.assertEqual(
            data["image"]["tag"], "vastai/vllm:v0.27.1-cuda-12.9"
        )
        self.assertEqual(data["image"]["digest"], "sha256:FILL_AFTER_FIRST_PULL")
        self.assertEqual(data["image"]["vllm_version"], "0.27.1")
        self.assertEqual(data["image"]["cuda"], "12.9")
        self.assertEqual(data["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(data["model"]["max_model_len"])
        self.assertNotIn("slo", data)
        self.assertNotIn("probe", data)
        self.assertNotIn("trace", data)
        self.assertNotIn("window", data)
        self.assertNotIn("noise", data)

    def test_shell_exports_match_run_sh_names(self) -> None:
        env = shell_exports(load_pin(DEFAULT_PIN_PATH))
        self.assertEqual(
            set(env),
            {
                "MODEL",
                "REVISION",
                "DTYPE",
                "MAX_MODEL_LEN",
            },
        )
        self.assertEqual(env["MODEL"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(env["DTYPE"], "bfloat16")
        self.assertEqual(env["MAX_MODEL_LEN"], "")
        self.assertEqual(env["REVISION"], "")
        self.assertTrue(DEFAULT_PIN_PATH.is_file())
        self.assertEqual(DEFAULT_PIN_PATH.name, "pin.yaml")
        self.assertEqual(DEFAULT_PIN_PATH.parent.name, "config")
        self.assertFalse((Path(DEFAULT_PIN_PATH).parent / "freeze.txt").exists())
        self.assertFalse((Path(DEFAULT_PIN_PATH).parent / "image.txt").exists())


class TestWriteMaxModelLen(unittest.TestCase):
    """write_max_model_len updates a pin.yaml copy in place."""

    def test_write_max_model_len_sets_key(self) -> None:
        """Copy pin.yaml, write 8192, assert the key and other fields stay."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.yaml"
            path.write_text(DEFAULT_PIN_PATH.read_text())
            write_max_model_len(path, 8192)
            data = load_pin(path)
            self.assertEqual(data["model"]["max_model_len"], 8192)
            self.assertEqual(data["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(data["model"]["dtype"], "bfloat16")
            self.assertEqual(
                data["image"]["tag"], "vastai/vllm:v0.27.1-cuda-12.9"
            )
            self.assertIn(
                "# See model.max_model_len in RUNBOOK.md", path.read_text()
            )

    def test_write_max_model_len_overwrites_existing(self) -> None:
        """A second write replaces the length without dropping other keys."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.yaml"
            path.write_text(DEFAULT_PIN_PATH.read_text())
            write_max_model_len(path, 4096)
            write_max_model_len(path, 16384)
            data = load_pin(path)
            self.assertEqual(data["model"]["max_model_len"], 16384)
            self.assertEqual(data["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")
            self.assertIsNone(data["model"]["revision"])


if __name__ == "__main__":
    unittest.main()
