#!/usr/bin/env python3
"""Local tests for gateway_outstanding drain check."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from goodput_lab.gateway_outstanding import (
    assert_drained,
    main,
    outstanding_from_text,
)

ZERO_BODY = (
    "# TYPE gateway_outstanding gauge\n"
    "gateway_outstanding 0\n"
    "# TYPE gateway_accepted_total counter\n"
    "gateway_accepted_total 12\n"
)
WEDGE_BODY = "# TYPE gateway_outstanding gauge\ngateway_outstanding 2\n"


class TestOutstandingFromText(unittest.TestCase):
    """Parse the gateway_outstanding gauge from Prometheus text."""

    def test_zero(self) -> None:
        """Idle gateway reports 0."""
        self.assertEqual(outstanding_from_text(ZERO_BODY), 0)

    def test_nonzero(self) -> None:
        """Wedged slot is a positive gauge."""
        self.assertEqual(outstanding_from_text(WEDGE_BODY), 2)

    def test_missing_raises(self) -> None:
        """A body without the gauge raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            outstanding_from_text("gateway_accepted_total 3\n")
        self.assertIn("gateway_outstanding", str(ctx.exception))


class TestAssertDrained(unittest.TestCase):
    """End-of-arm outstanding must be 0."""

    def test_zero_ok(self) -> None:
        """Drained returns 0."""
        self.assertEqual(assert_drained(ZERO_BODY), 0)

    def test_nonzero_raises(self) -> None:
        """Nonzero outstanding is a wedged slot."""
        with self.assertRaises(ValueError) as ctx:
            assert_drained(WEDGE_BODY)
        self.assertIn("discard arm", str(ctx.exception))


class TestMain(unittest.TestCase):
    """CLI reads stdin and exits 1 when outstanding is not 0."""

    def test_zero_prints_and_exits_0(self) -> None:
        """Drained stdin prints the gauge and exits 0."""
        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO(ZERO_BODY)), patch(
            "sys.stdout", stdout
        ):
            code = main([])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "gateway_outstanding 0\n")

    def test_nonzero_exits_1(self) -> None:
        """Wedged stdin prints the error and exits 1."""
        stderr = io.StringIO()
        with patch("sys.stdin", io.StringIO(WEDGE_BODY)), patch(
            "sys.stderr", stderr
        ):
            code = main([])
        self.assertEqual(code, 1)
        self.assertIn("gateway_outstanding 2", stderr.getvalue())

    def test_empty_stdin_exits_1(self) -> None:
        """Missing gauge on empty stdin is a failed drain check."""
        stderr = io.StringIO()
        with patch("sys.stdin", io.StringIO("")), patch("sys.stderr", stderr):
            code = main([])
        self.assertEqual(code, 1)
        self.assertIn("missing gateway_outstanding", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
