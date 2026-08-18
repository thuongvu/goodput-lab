"""
Stage              | Call it          | Type             | File
Whole CSV          | log              | -                | request_window.py
One CSV line       | recorded request | RecordedRequest  | request_window.py
Time range         | window           | [start, end)     | --window-sec
Rows in that range | request window   | RequestWindow    | request_window.py (from_csv)
Plan arrival_scale / stride / max_runtime | window policy | -          | window_policy.py
Replay JSONL       | timed_trace      | TimedTraceRecord | timed_trace.py

from_csv, then consult policy, then write timed_trace. Policy recommends; timed_trace applies.

Fill a RequestWindow from a local Azure Conversation CSV. Stdlib only (runs before any GPU).

The file is a trace of LLM inference invocations:
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
Each row is a request with an arrival time, input size, and output size:
  https://arxiv.org/html/2311.18677v2

RecordedRequest: one logged request (one CSV line). Not a live vLLM Request.
                  arrival_time / input_length / output_length match vLLM.
window:           time interval [start, end) on the Azure log
                  --window-sec N alone = first N seconds of the CSV (first TIMESTAMP)
                  --start/--end = a specific slice once you know it
                  CSV must be sorted by TIMESTAMP (from_csv stops at first arrival >= end).
RequestWindow:    the recorded requests in that window, not the whole log
                  (length-0 completions omitted).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

AZURE_CONV_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv"
)

DEFAULT_CSV = (
    Path(__file__).resolve().parent / "data" / "AzureLLMInferenceTrace_conv_1week.csv"
)


@dataclass(frozen=True)
class RecordedRequest:
    """One logged request (one CSV line). Not a live vLLM Request.

    Field names match vLLM SampleRequest / timed_trace. Parsed from Azure
    TIMESTAMP, ContextTokens, GeneratedTokens.
    """

    index: int  # 0-based data row in the CSV (not counting header)
    arrival_time: datetime  # Azure TIMESTAMP is invocation time; vLLM Request.arrival_time
    input_length: int  # Azure ContextTokens
    output_length: int  # Azure GeneratedTokens

    @property
    def total_tokens(self) -> int:
        """input_length + output_length."""
        return self.input_length + self.output_length


@dataclass(frozen=True)
class RequestWindow:
    """The recorded requests in a time window [start, end) of the Azure log. Not the whole log.

    Completions of length 0 are omitted because there is nothing to generate.
    Long prompts are kept here. Fitting to a model context length happens in timed_trace.
    """

    rows: list[RecordedRequest]
    scanned: int
    dropped_gen0: int


def from_csv(
    path: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    max_rows: int | None = None,
) -> RequestWindow:
    """Read a local Azure Conversation CSV at path and return a RequestWindow for [start, end).

    The CSV must be sorted by TIMESTAMP: rows before start are skipped, then
    reading stops at the first arrival_time >= end.

    Does not download. Omits completions of length 0 (nothing to generate). Long
    prompts are kept here. Fitting to a model context length happens in timed_trace.
    """
    rows: list[RecordedRequest] = []
    scanned = 0
    dropped_gen0 = 0
    for row in _iter_azure_csv(path):
        if start is not None and row.arrival_time < start:
            continue
        if end is not None and row.arrival_time >= end:
            break
        scanned += 1
        if row.output_length == 0:
            dropped_gen0 += 1
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return RequestWindow(rows=rows, scanned=scanned, dropped_gen0=dropped_gen0)


def fits_max_model_len(row: RecordedRequest, max_model_len: int) -> bool:
    """Non-empty prompt whose prompt+output fits vLLM max_model_len."""
    return row.input_length > 0 and row.total_tokens <= max_model_len


def check_window_args(
    start: str | None,
    end: str | None,
    window_sec: float | None,
) -> None:
    """CLI window flags that can be checked before opening the CSV."""
    if end and window_sec is not None:
        raise ValueError("pass only one of --end and --window-sec")


def first_arrival_time(path: Path) -> datetime:
    """First TIMESTAMP in the CSV (file order). Empty file -> ValueError."""
    for row in _iter_azure_csv(path):
        return row.arrival_time
    raise ValueError("no rows in {}".format(path))


def window_bounds(
    start: str | None,
    end: str | None,
    window_sec: float | None,
    *,
    default_start: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Turn --start/--end/--window-sec into [start, end) for from_csv.

    --window-sec without --start uses default_start (CSV first TIMESTAMP):
    first N seconds of the file. Pass only one of --end and --window-sec.
    """
    check_window_args(start, end, window_sec)
    t0 = _parse_ts(start) if start else default_start
    if end:
        return t0, _parse_ts(end)
    if window_sec is not None:
        if t0 is None:
            raise ValueError(
                "--window-sec without --start needs the CSV's first TIMESTAMP"
            )
        return t0, t0 + timedelta(seconds=window_sec)
    return t0, None


def resolve_window_bounds(
    csv_path: Path,
    start: str | None,
    end: str | None,
    window_sec: float | None,
) -> tuple[datetime | None, datetime | None]:
    """window_bounds, peeking the CSV's first TIMESTAMP when --window-sec has no --start."""
    default_start = None
    if window_sec is not None and not start:
        default_start = first_arrival_time(csv_path)
    return window_bounds(start, end, window_sec, default_start=default_start)


def download_azure_csv(dest: Path = DEFAULT_CSV) -> Path:
    """Download the Azure Conversation CSV if dest is missing or empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urlretrieve(AZURE_CONV_URL, dest)
    return dest


def _iter_azure_csv(path: Path) -> Iterator[RecordedRequest]:
    """Yield RecordedRequest values from a Conversation CSV, in file order."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("no header in {}".format(path))
        fields = {name.strip(): name for name in reader.fieldnames}
        for required in ("TIMESTAMP", "ContextTokens", "GeneratedTokens"):
            if required not in fields:
                raise ValueError(
                    "{} missing {}; got {}".format(
                        path, required, reader.fieldnames
                    )
                )
        ts_k = fields["TIMESTAMP"]
        ctx_k = fields["ContextTokens"]
        gen_k = fields["GeneratedTokens"]
        for i, row in enumerate(reader):
            yield RecordedRequest(
                index=i,
                arrival_time=_parse_ts(row[ts_k]),
                input_length=int(row[ctx_k]),
                output_length=int(row[gen_k]),
            )


def _parse_ts(raw: str) -> datetime:
    """Parse Azure TIMESTAMP / --start/--end ISO strings to UTC datetime."""
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
