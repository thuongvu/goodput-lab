"""Load config/pin.yaml. Print export KEY=value for run.sh when invoked as __main__."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIN_PATH = _REPO_ROOT / "config" / "pin.yaml"


def load_pin(path: Path | None = None) -> dict[str, Any]:
    """yaml.safe_load pin.yaml. Missing/empty file -> {}."""
    p = Path(path) if path is not None else DEFAULT_PIN_PATH
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text())
    return data if isinstance(data, dict) else {}


def write_max_model_len(path: Path, value: int) -> None:
    """Set model.max_model_len in pin.yaml. Other keys and comments stay."""
    p = Path(path)
    lines = p.read_text().splitlines(keepends=True)
    found = False
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("max_model_len:"):
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            out.append(
                "{}max_model_len: {}{}".format(indent, int(value), newline)
            )
            found = True
        else:
            out.append(line)
    if not found:
        raise ValueError("no max_model_len key in {}".format(p))
    p.write_text("".join(out))


def _env(value: Any) -> str:
    """Flatten a YAML scalar to a shell string. None -> empty (same as KEY=)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def shell_exports(data: dict[str, Any]) -> dict[str, str]:
    """Map pin.yaml model fields onto the env names run.sh evals."""
    model = data.get("model") or {}
    raw = {
        "MODEL": model.get("name"),
        "REVISION": model.get("revision"),
        "DTYPE": model.get("dtype"),
        "MAX_MODEL_LEN": model.get("max_model_len"),
    }
    return {key: _env(val) for key, val in raw.items()}


def main(argv: list[str] | None = None) -> int:
    """Print `export KEY=value` lines for eval in run.sh."""
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PIN_PATH
    for key, val in shell_exports(load_pin(path)).items():
        print("export {}={}".format(key, shlex.quote(val)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
