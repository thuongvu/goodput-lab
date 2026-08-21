#!/usr/bin/env python3
"""Launch settings from a run dir for the next vLLM start.

pin.yaml is the model and image identity (name, dtype, tag, digest). 
serve.cmd is the vLLM launch line, parsed into argv, host, port, and prefix-caching. 
serve.log is the engine log; max_num_seqs, max_num_batched_tokens, and gpu_memory_utilization
are LoggedValue records. 
vllm_default_not_passed means the flag was missing from serve.log.

Writes serve-settings.json.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from goodput_lab.config import load_pin
from goodput_lab.run_metadata import RunMetadata
from goodput_lab.run_metadata import from_run_dir as metadata_from_run_dir

SERVE_SETTINGS_FILENAME = "serve-settings.json"
STOCK_SERVE_DEFAULT = "vllm_default_not_passed"


# SERVE SETTINGS
# Nested records for serve-settings.json.


@dataclass(frozen=True)
class ModelPin:
    """Model identity copied from the run dir pin.yaml."""

    name: object
    dtype: object
    max_model_len: object
    revision: object


@dataclass(frozen=True)
class ImagePin:
    """Image identity copied from the run dir pin.yaml."""

    tag: object
    digest: object
    vllm_version: object
    cuda: object


@dataclass(frozen=True)
class ServeProcess:
    """serve.cmd text, argv, and flags that were actually passed. What to launch next time."""

    cmd: str
    argv: list[str]
    host: str | None
    port: int | str | None
    no_enable_prefix_caching: bool


@dataclass(frozen=True)
class LoggedValue:
    """Number parsed from serve.log, or vllm_default_not_passed when the flag is missing."""

    source: str
    value: int | float | None = None


@dataclass(frozen=True)
class ServeSettings:
    """Launch settings captured from this run, for next time: pin, serve.cmd, and 
    serve.log engine settings max_num_seqs, max_num_batched_tokens, and gpu_memory_utilization.
    """

    model: ModelPin
    image: ImagePin
    serve: ServeProcess
    metadata: RunMetadata
    max_num_seqs: LoggedValue
    max_num_batched_tokens: LoggedValue
    gpu_memory_utilization: LoggedValue

    def to_dict(self) -> dict:
        """JSON-ready dict for serve-settings.json. Omits value when the flag is missing from serve.log."""
        payload = asdict(self)
        for key in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
        ):
            if payload[key]["value"] is None:
                del payload[key]["value"]
        return payload


# LOAD
# pin.yaml, serve.cmd, metadata.json, serve.log -> ServeSettings.


def from_run_dir(run_dir: Path) -> ServeSettings:
    """Read pin.yaml, serve.cmd, metadata, and serve.log from a run dir."""
    pin = load_pin(run_dir / "pin.yaml")
    serve_log_path = run_dir / "serve.log"
    serve_log = serve_log_path.read_text() if serve_log_path.is_file() else ""
    return ServeSettings(
        model=_model_pin(pin.get("model") or {}),
        image=_image_pin(pin.get("image") or {}),
        serve=_serve_process(run_dir),
        metadata=metadata_from_run_dir(run_dir),
        max_num_seqs=_from_serve_log(serve_log, "max_num_seqs", as_int=True),
        max_num_batched_tokens=_from_serve_log(
            serve_log, "max_num_batched_tokens", as_int=True
        ),
        gpu_memory_utilization=_from_serve_log(
            serve_log, "gpu_memory_utilization", as_int=False
        ),
    )


# pin.yaml


def _model_pin(model: dict) -> ModelPin:
    """Model fields from a pin.yaml mapping."""
    return ModelPin(
        name=model.get("name"),
        dtype=model.get("dtype"),
        max_model_len=model.get("max_model_len"),
        revision=model.get("revision"),
    )


def _image_pin(image: dict) -> ImagePin:
    """Image fields from a pin.yaml mapping."""
    return ImagePin(
        tag=image.get("tag"),
        digest=image.get("digest"),
        vllm_version=image.get("vllm_version"),
        cuda=image.get("cuda"),
    )


# serve.cmd


def _serve_process(run_dir: Path) -> ServeProcess:
    """Parse serve.cmd from a run dir into argv and host/port flags."""
    serve_cmd_path = run_dir / "serve.cmd"
    serve_cmd_text = serve_cmd_path.read_text() if serve_cmd_path.is_file() else ""
    argv = _parse_serve_cmd(serve_cmd_text)
    return ServeProcess(
        cmd=serve_cmd_text.strip(),
        argv=argv,
        host=_flag_value(argv, "--host"),
        port=_parse_port(_flag_value(argv, "--port")),
        no_enable_prefix_caching="--no-enable-prefix-caching" in argv,
    )


def _parse_serve_cmd(text: str) -> list[str]:
    """Split the first serve.cmd line into argv."""
    if not text.strip():
        return []
    line = text.strip().splitlines()[0]
    if line.startswith("serve:"):
        line = line[len("serve:") :].strip()
    return shlex.split(line)


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Looks up a flag's argument on the parsed serve.cmd argv."""
    prefix = flag + "="
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _parse_port(raw: str | None) -> int | str | None:
    """The --port flag from argv as an integer"""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


# serve.log


def _from_serve_log(text: str, key: str, *, as_int: bool) -> LoggedValue:
    """Parse key=value or 'key': value from serve.log."""
    if not text:
        return LoggedValue(source=STOCK_SERVE_DEFAULT)
    pattern = r"""['\"]?{}['\"]?\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)""".format(
        re.escape(key)
    )
    match = re.search(pattern, text)
    if match is None:
        return LoggedValue(source=STOCK_SERVE_DEFAULT)
    raw = match.group(1)
    value: int | float = int(raw) if as_int else float(raw)
    if as_int and "." in raw:
        value = int(float(raw))
    return LoggedValue(value=value, source="serve.log")


# WRITE
# serve-settings.json into the run dir.


def write(run_dir: Path, settings: ServeSettings) -> Path:
    """Write serve-settings.json into run_dir. results.py CLI also calls this."""
    path = run_dir / SERVE_SETTINGS_FILENAME
    path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n")
    return path
