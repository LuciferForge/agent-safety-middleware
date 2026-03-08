"""Tests for Flask middleware — tests extraction and guard logic without Flask."""

import pytest
from unittest.mock import MagicMock
from agent_safety_middleware.flask_middleware import FlaskAgentSafety


class TestFlaskAgentSafety:

    def test_init_registers_hooks(self):
        app = MagicMock()
        app.extensions = {}
        safety = FlaskAgentSafety(app)
        app.before_request.assert_called_once()
        app.after_request.assert_called_once()
        assert "agent_safety" in app.extensions

    def test_init_app_deferred(self):
        safety = FlaskAgentSafety()
        app = MagicMock()
        app.extensions = {}
        safety.init_app(app)
        app.before_request.assert_called_once()

    def test_add_headers(self):
        safety = FlaskAgentSafety()
        resp = MagicMock()
        resp.headers = {}
        result = safety._add_headers(resp)
        assert result.headers["X-Safety-Checked"] == "true"
        assert "X-Safety-Requests" in result.headers

    def test_cost_remaining_header(self):
        safety = FlaskAgentSafety(max_cost_per_session=10.00)
        resp = MagicMock()
        resp.headers = {}
        result = safety._add_headers(resp)
        assert "X-Safety-Cost-Remaining" in result.headers


class TestFlaskTextExtraction:

    def test_json_body(self):
        safety = FlaskAgentSafety()
        req = MagicMock()
        req.get_json.return_value = {"prompt": "hello world"}
        text = safety._extract_text(req)
        assert "hello world" in text

    def test_nested_messages(self):
        safety = FlaskAgentSafety()
        req = MagicMock()
        req.get_json.return_value = {
            "messages": [{"role": "user", "content": "test message"}]
        }
        text = safety._extract_text(req)
        assert "test message" in text

    def test_raw_body_fallback(self):
        safety = FlaskAgentSafety()
        req = MagicMock()
        req.get_json.return_value = None
        req.get_data.return_value = "raw text body"
        text = safety._extract_text(req)
        assert "raw text body" in text

    def test_multiple_fields(self):
        safety = FlaskAgentSafety()
        req = MagicMock()
        req.get_json.return_value = {"prompt": "hello", "query": "world"}
        text = safety._extract_text(req)
        assert "hello" in text
        assert "world" in text

    def test_empty_body(self):
        safety = FlaskAgentSafety()
        req = MagicMock()
        req.get_json.return_value = None
        req.get_data.return_value = ""
        text = safety._extract_text(req)
        assert text == ""

    def test_deep_nesting_no_crash(self):
        safety = FlaskAgentSafety()
        deep = {"content": {"content": {"content": {"content": {"content": {"content": "deep"}}}}}}
        req = MagicMock()
        req.get_json.return_value = deep
        text = safety._extract_text(req)
        assert isinstance(text, str)
