#!/usr/bin/env python3
"""Unit tests for run_metadata: write keys, indent, round-trip."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from goodput_lab.run_metadata import RunMetadata, from_run_dir, write


class TestRunMetadata(unittest.TestCase):
    def test_write_keys_and_indent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta = RunMetadata(
                git_head="abc123", chunk_hash_size=16, mode="loaded"
            )
            path = write(run_dir, meta)
            text = path.read_text()
            payload = json.loads(text)
            self.assertEqual(
                list(payload), ["git_head", "chunk_hash_size", "mode"]
            )
            self.assertEqual(payload["git_head"], "abc123")
            self.assertEqual(payload["chunk_hash_size"], 16)
            self.assertEqual(payload["mode"], "loaded")
            self.assertEqual(
                text,
                json.dumps(
                    {
                        "git_head": "abc123",
                        "chunk_hash_size": 16,
                        "mode": "loaded",
                    },
                    indent=2,
                )
                + "\n",
            )
            self.assertEqual(from_run_dir(run_dir), meta)


if __name__ == "__main__":
    unittest.main()
