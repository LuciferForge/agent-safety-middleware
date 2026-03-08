"""Decorator for protecting individual endpoints.

Usage:
    from agent_safety_middleware import safe_endpoint

    @app.post("/chat")
    @safe_endpoint(injection_threshold=5, max_cost_per_request=0.50)
    async def chat(request):
        ...
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable, Optional

from agent_safety_middleware.guard import SafetyGuard


def safe_endpoint(
    injection_threshold: float = 5.0,
    max_cost_per_request: Optional[float] = None,
    on_injection: str = "block",
    extract_field: str = "prompt",
    guard: Optional[SafetyGuard] = None,
) -> Callable:
    """Decorator that adds safety checks to an individual endpoint function.

    Works with both sync and async functions. Extracts text from the first
    argument (if string) or from the request body.

    Args:
        injection_threshold: Score threshold for blocking.
        max_cost_per_request: Max cost per request in USD.
        on_injection: "block" (return error), "flag" (add to kwargs), "log" (print warning).
        extract_field: Field name to extract from request body for scanning.
        guard: Optional shared SafetyGuard instance. Creates new one if None.
    """

    def decorator(func: Callable) -> Callable:
        _guard = guard or SafetyGuard(
            injection_threshold=injection_threshold,
            max_cost_per_request=max_cost_per_request,
            on_injection=on_injection,
            trace_name=f"endpoint-{func.__name__}",
        )

        import inspect
        sig = inspect.signature(func)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        has_safety_param = "_safety_result" in sig.parameters

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            text = _extract_from_args(args, kwargs, extract_field)
            if text:
                result = _guard.check(text)
                if not result.safe:
                    return _blocked_response(result)
                if accepts_kwargs or has_safety_param:
                    kwargs["_safety_result"] = result
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            text = _extract_from_args(args, kwargs, extract_field)
            if text:
                result = _guard.check(text)
                if not result.safe:
                    return _blocked_response(result)
                if accepts_kwargs or has_safety_param:
                    kwargs["_safety_result"] = result
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _extract_from_args(args: tuple, kwargs: dict, field: str) -> str:
    """Try to extract scannable text from function arguments."""
    # Check kwargs first
    if field in kwargs:
        val = kwargs[field]
        if isinstance(val, str):
            return val

    # Check for common request objects
    for arg in args:
        # String argument
        if isinstance(arg, str):
            return arg
        # Dict with the target field
        if isinstance(arg, dict) and field in arg:
            val = arg[field]
            if isinstance(val, str):
                return val

    return ""


def _blocked_response(result: Any) -> dict:
    """Return a standard blocked response dict."""
    return {
        "error": "Request blocked by safety guard",
        "reason": result.blocked_reason,
        "injection_score": result.injection_score,
    }
