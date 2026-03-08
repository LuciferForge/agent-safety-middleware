"""Flask middleware for agent safety.

Usage:
    from flask import Flask
    from agent_safety_middleware import FlaskAgentSafety

    app = Flask(__name__)
    FlaskAgentSafety(app)

    # With options:
    FlaskAgentSafety(app, injection_threshold=10, max_cost_per_session=5.00)
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent_safety_middleware.guard import SafetyGuard


class FlaskAgentSafety:
    """Flask extension that adds safety checks to incoming requests.

    Registers a before_request hook that scans POST/PUT/PATCH bodies
    for injection attacks and enforces cost limits.
    """

    def __init__(
        self,
        app: Any = None,
        injection_threshold: float = 5.0,
        max_cost_per_request: Optional[float] = None,
        max_cost_per_session: Optional[float] = None,
        enable_tracing: bool = True,
        on_injection: str = "block",
        block_response_code: int = 403,
        skip_paths: Optional[list[str]] = None,
        extract_fields: Optional[list[str]] = None,
    ):
        self.block_response_code = block_response_code
        self.skip_paths = skip_paths or ["/health", "/healthz", "/metrics"]
        self.extract_fields = extract_fields or [
            "prompt", "message", "content", "input", "query",
            "text", "user_message", "question", "messages",
        ]
        self.guard = SafetyGuard(
            injection_threshold=injection_threshold,
            max_cost_per_request=max_cost_per_request,
            max_cost_per_session=max_cost_per_session,
            enable_tracing=enable_tracing,
            on_injection=on_injection,
            trace_name="flask-safety",
        )

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Any) -> None:
        """Register the safety middleware with a Flask app."""
        app.before_request(self._check_request)
        app.after_request(self._add_headers)
        app.extensions["agent_safety"] = self

    def _check_request(self) -> Any:
        """Before-request hook: scan body for injections, enforce cost limits."""
        try:
            from flask import request, jsonify, make_response
        except ImportError:
            return None

        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None

        if any(request.path.startswith(skip) for skip in self.skip_paths):
            return None

        text = self._extract_text(request)
        if not text:
            return None

        result = self.guard.check(text)

        if not result.safe:
            response = make_response(
                jsonify({
                    "error": "Request blocked by safety middleware",
                    "reason": result.blocked_reason,
                    "injection_score": result.injection_score,
                }),
                self.block_response_code,
            )
            response.headers["X-Safety-Blocked"] = "true"
            return response

        return None

    def _add_headers(self, response: Any) -> Any:
        """After-request hook: add safety headers."""
        response.headers["X-Safety-Checked"] = "true"
        response.headers["X-Safety-Requests"] = str(self.guard.request_count)
        if self.guard.max_cost_per_session:
            remaining = self.guard.max_cost_per_session - self.guard.total_cost
            response.headers["X-Safety-Cost-Remaining"] = f"{remaining:.4f}"
        return response

    def _extract_text(self, request: Any) -> str:
        """Extract scannable text from Flask request."""
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data is None:
            # Try form data or raw body
            body = request.get_data(as_text=True)
            return body[:10000] if body else ""

        texts: list[str] = []
        self._walk_extract(data, texts, depth=0)
        return " ".join(texts)[:10000]

    def _walk_extract(self, data: Any, texts: list, depth: int = 0) -> None:
        """Recursively extract text fields from request data."""
        if depth > 5:
            return

        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.lower() in self.extract_fields:
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            self._walk_extract(item, texts, depth + 1)
                    elif isinstance(value, dict):
                        self._walk_extract(value, texts, depth + 1)
                elif isinstance(value, (dict, list)):
                    self._walk_extract(value, texts, depth + 1)
        elif isinstance(data, list):
            for item in data:
                self._walk_extract(item, texts, depth + 1)
