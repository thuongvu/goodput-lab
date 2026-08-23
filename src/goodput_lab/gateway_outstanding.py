#!/usr/bin/env python3
"""Parse gateway_outstanding gauge from Prometheus text and fail unless it's 0."""

from __future__ import annotations

import argparse
import sys

SERIES = "gateway_outstanding"


def outstanding_from_text(body: str) -> int:
    """gateway_outstanding gauge from Prometheus text."""
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if name == SERIES or name.startswith(SERIES + "{"):
            return int(float(value))
    raise ValueError("missing {}".format(SERIES))


def assert_drained(body: str) -> int:
    """Fail unless outstanding is 0. Returns 0 on success."""
    outstanding = outstanding_from_text(body)
    if outstanding != 0:
        raise ValueError(
            "{} {} (wedged slot; discard arm)".format(SERIES, outstanding)
        )
    return outstanding


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: Prometheus text on stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read stdin; exit 0 if outstanding is 0, else 1."""
    parse_args(argv)
    body = sys.stdin.read()
    try:
        outstanding = assert_drained(body)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 1
    print("{} {}".format(SERIES, outstanding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
