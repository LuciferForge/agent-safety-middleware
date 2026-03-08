"""FastAPI/ASGI middleware for agent safety.

Usage:
    from fastapi import FastAPI
    from agent_safety_middleware import AgentSafetyMiddleware

    app = FastAPI()
    app.add_middleware(AgentSafetyMiddleware)

    # With options:
    app.add_middleware(
        AgentSafetyMiddleware,
        injection_threshold=10,
        max_cost_per_session=5.00,
        block_response_code=403,
    )
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agent_safety_middleware.guard import SafetyGuard


class AgentSafetyMiddleware:
    """ASGI middleware that scans request bodies for injection attacks and enforces cost limits.

    Intercepts POST/PUT/PATCH requests, extracts the body, runs SafetyGuard checks,
    and returns a 403 if the input is blocked. Adds X-Safety-* headers to all responses.
    """

    def __init__(
        self,
        app: Any,
        injection_threshold: float = 5.0,
        max_cost_per_request: Optional[float] = None,
        max_cost_per_session: Optional[float] = None,
        enable_tracing: bool = True,
        on_injection: str = "block",
        block_response_code: int = 403,
        scan_paths: Optional[list[str]] = None,
        skip_paths: Optional[list[str]] = None,
        extract_fields: Optional[list[str]] = None,
    ):
        self.app = app
        self.block_response_code = block_response_code
        self.scan_paths = scan_paths
        self.skip_paths = skip_paths or ["/health", "/healthz", "/metrics", "/docs", "/openapi.json"]
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
            trace_name="fastapi-safety",
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Skip non-mutating methods and excluded paths
        if method in ("GET", "HEAD", "OPTIONS"):
            await self._pass_with_headers(scope, receive, send)
            return

        if any(path.startswith(skip) for skip in self.skip_paths):
            await self._pass_with_headers(scope, receive, send)
            return

        if self.scan_paths and not any(path.startswith(sp) for sp in self.scan_paths):
            await self._pass_with_headers(scope, receive, send)
            return

        # Read the body
        body = b""
        request_complete = False

        async def receive_wrapper() -> dict:
            nonlocal body, request_complete
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    request_complete = True
            return message

        # Buffer the full body
        while not request_complete:
            await receive_wrapper()

        # Extract text to scan
        text_to_scan = self._extract_text(body)

        if text_to_scan:
            result = self.guard.check(text_to_scan)

            if not result.safe:
                # Block the request
                response_body = json.dumps({
                    "error": "Request blocked by safety middleware",
                    "reason": result.blocked_reason,
                    "injection_score": result.injection_score,
                }).encode()

                await send({
                    "type": "http.response.start",
                    "status": self.block_response_code,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"x-safety-blocked", b"true"],
                        [b"x-safety-reason", result.blocked_reason.encode()[:200] if result.blocked_reason else b""],
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": response_body,
                })
                return

        # Pass through — replay the buffered body
        body_sent = False

        async def replay_receive() -> dict:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self._pass_with_headers(scope, replay_receive, send)

    async def _pass_with_headers(self, scope: dict, receive: Callable, send: Callable) -> None:
        """Pass request through, adding safety headers to the response."""
        headers_added = False

        async def send_wrapper(message: dict) -> None:
            nonlocal headers_added
            if message["type"] == "http.response.start" and not headers_added:
                headers_added = True
                headers = list(message.get("headers", []))
                headers.append([b"x-safety-checked", b"true"])
                headers.append([
                    b"x-safety-requests",
                    str(self.guard.request_count).encode(),
                ])
                if self.guard.max_cost_per_session:
                    headers.append([
                        b"x-safety-cost-remaining",
                        f"{self.guard.max_cost_per_session - self.guard.total_cost:.4f}".encode(),
                    ])
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _extract_text(self, body: bytes) -> str:
        """Extract scannable text from request body."""
        if not body:
            return ""

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON — scan raw body as text
            try:
                return body.decode("utf-8", errors="ignore")[:10000]
            except Exception:
                return ""

        texts = []
        self._walk_extract(data, texts, depth=0)
        return " ".join(texts)[:10000]

    def _walk_extract(self, data: Any, texts: list, depth: int = 0) -> None:
        """Recursively extract text fields from JSON data."""
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
