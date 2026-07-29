"""
tests/test_utils.py
====================
Unit tests for utility helper functions.
"""

import pytest

from utils.helper import (
    Timer,
    build_error_response,
    build_success_response,
    generate_request_id,
    sanitize_text,
    truncate_text,
    utcnow_iso,
)


class TestSanitizeText:
    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert sanitize_text("hello   world") == "hello world"

    def test_handles_newlines(self):
        assert sanitize_text("hello\nworld\t!") == "hello world !"

    def test_empty_string(self):
        assert sanitize_text("") == ""


class TestTruncateText:
    def test_no_truncation_within_limit(self):
        text = "a" * 100
        assert truncate_text(text, max_length=200) == text

    def test_truncates_at_limit(self):
        text = "a" * 600
        result = truncate_text(text, max_length=512)
        assert len(result) == 512
        assert result.endswith("...")

    def test_exactly_at_limit(self):
        text = "a" * 512
        assert truncate_text(text, max_length=512) == text


class TestGenerateRequestId:
    def test_returns_string(self):
        assert isinstance(generate_request_id(), str)

    def test_unique_ids(self):
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestUtcnowIso:
    def test_returns_string(self):
        assert isinstance(utcnow_iso(), str)

    def test_contains_timezone(self):
        ts = utcnow_iso()
        assert "+" in ts or ts.endswith("Z")


class TestTimer:
    def test_elapsed_ms_is_positive(self):
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms > 0

    def test_elapsed_seconds_is_positive(self):
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_seconds > 0


class TestResponseBuilders:
    def test_success_response_has_required_keys(self):
        resp = build_success_response(data={"key": "value"})
        assert resp["success"] is True
        assert "data" in resp
        assert "timestamp" in resp
        assert "request_id" in resp

    def test_error_response_has_required_keys(self):
        resp = build_error_response(error="Test error")
        assert resp["success"] is False
        assert resp["error"] == "Test error"
        assert "timestamp" in resp
        assert "request_id" in resp
