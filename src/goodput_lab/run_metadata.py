#!/usr/bin/env python3
"""Write metadata.json into a run dir (git_head, chunk_hash_size, mode)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class RunMetadata:
    """Fields written to a run dir's metadata.json.

    git_head: git rev-parse HEAD, or "uncommitted" if git fails.
    chunk_hash_size: vLLM timed-trace chunk hash size (run.sh pin).
    mode: discover or loaded.
    """

    git_head: str
    chunk_hash_size: int
    mode: str


def write(run_dir: Path, metadata: RunMetadata) -> Path:
    """Write metadata.json into run_dir. Returns the path written."""
    path = run_dir / METADATA_FILENAME
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    return path


def from_run_dir(run_dir: Path) -> RunMetadata:
    """Read metadata.json from a run dir."""
    payload = json.loads((run_dir / METADATA_FILENAME).read_text())
    return RunMetadata(
        git_head=payload["git_head"],
        chunk_hash_size=int(payload["chunk_hash_size"]),
        mode=payload["mode"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: run dir, then git_head, chunk_hash_size, mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("git_head")
    parser.add_argument("chunk_hash_size", type=int)
    parser.add_argument("mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write metadata.json into run_dir from the four CLI args."""
    args = parse_args(argv)
    write(
        args.run_dir,
        RunMetadata(
            git_head=args.git_head,
            chunk_hash_size=args.chunk_hash_size,
            mode=args.mode,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
