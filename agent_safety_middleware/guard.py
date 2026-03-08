"""Core safety guard — injection scanning, cost tracking, decision tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_shield import PromptScanner
from ai_trace import Tracer


@dataclass
class SafetyResult:
    """Result of a safety check."""

    safe: bool
    injection_flagged: bool = False
    injection_score: float = 0.0
    injection_matches: list[dict] = field(default_factory=list)
    cost_blocked: bool = False
    cost_remaining: float = 0.0
    trace_id: Optional[str] = None
    blocked_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "injection_flagged": self.injection_flagged,
            "injection_score": self.injection_score,
            "injection_matches": [m.get("name", "") for m in self.injection_matches],
            "cost_blocked": self.cost_blocked,
            "cost_remaining": self.cost_remaining,
            "trace_id": self.trace_id,
            "blocked_reason": self.blocked_reason,
        }


class SafetyGuard:
    """Unified safety guard combining injection scanning, cost tracking, and tracing.

    Args:
        injection_threshold: Risk score threshold for blocking (0-100). Default 5.
        max_cost_per_request: Max cost in USD per individual request. Default None.
        max_cost_per_session: Max total cost in USD across all requests. Default None.
        enable_tracing: Log decisions to ai-decision-tracer. Default True.
        trace_name: Name for the trace session. Default "agent-safety".
        on_injection: Action on detection. "block" (default), "flag", "log".
    """

    def __init__(
        self,
        injection_threshold: float = 5.0,
        max_cost_per_request: Optional[float] = None,
        max_cost_per_session: Optional[float] = None,
        enable_tracing: bool = True,
        trace_name: str = "agent-safety",
        on_injection: str = "block",
        **kwargs: Any,
    ):
        self.injection_threshold = injection_threshold
        self.max_cost_per_request = max_cost_per_request
        self.max_cost_per_session = max_cost_per_session
        self.enable_tracing = enable_tracing
        self.on_injection = on_injection
        self._total_cost = 0.0
        self._request_count = 0

        # Initialize scanner
        self._scanner = PromptScanner()

        # Initialize tracer
        self._tracer = None
        if enable_tracing:
            self._tracer = Tracer(agent=trace_name, auto_save=False)

    def check(self, text: str, estimated_cost: float = 0.0, metadata: Optional[dict] = None) -> SafetyResult:
        """Run all safety checks on input text."""
        result = SafetyResult(safe=True)
        step_data: dict[str, Any] = {"input_length": len(text), "estimated_cost": estimated_cost}
        if metadata:
            step_data["metadata"] = metadata

        # 1. Injection scan
        scan_result = self._scanner.scan(text)
        result.injection_score = scan_result.risk_score
        result.injection_matches = scan_result.matches

        if scan_result.risk_score >= self.injection_threshold:
            result.injection_flagged = True
            step_data["injection_score"] = scan_result.risk_score
            step_data["injection_matches"] = [m.get("name", "") for m in scan_result.matches]

            if self.on_injection == "block":
                result.safe = False
                result.blocked_reason = f"Injection detected (score: {scan_result.risk_score})"
                self._trace_step("blocked", step_data)
                return result

        # 2. Cost check
        if self.max_cost_per_request and estimated_cost > self.max_cost_per_request:
            result.safe = False
            result.cost_blocked = True
            result.blocked_reason = f"Cost ${estimated_cost:.4f} exceeds per-request limit ${self.max_cost_per_request:.4f}"
            self._trace_step("cost_blocked", step_data)
            return result

        if self.max_cost_per_session:
            if self._total_cost + estimated_cost > self.max_cost_per_session:
                result.safe = False
                result.cost_blocked = True
                result.blocked_reason = (
                    f"Session cost ${self._total_cost + estimated_cost:.4f} "
                    f"would exceed limit ${self.max_cost_per_session:.4f}"
                )
                self._trace_step("session_cost_blocked", step_data)
                return result
            result.cost_remaining = self.max_cost_per_session - self._total_cost - estimated_cost

        # Track cost and count
        self._total_cost += estimated_cost
        self._request_count += 1

        step_data["total_cost"] = self._total_cost
        step_data["request_number"] = self._request_count
        self._trace_step("allowed", step_data)

        if self._tracer:
            result.trace_id = self._tracer.agent

        return result

    def _trace_step(self, action: str, data: dict) -> None:
        """Log a decision step to the tracer."""
        if not self._tracer:
            return
        with self._tracer.step(f"safety_{action}", action=action, **data):
            pass

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def tracer(self) -> Optional[Tracer]:
        return self._tracer

    def get_trace(self) -> Optional[dict]:
        """Get the trace summary."""
        if self._tracer:
            return self._tracer.summary()
        return None
