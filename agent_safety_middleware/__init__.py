"""Agent Safety Middleware — one-line safety for AI agent APIs.

Usage (FastAPI):
    from agent_safety_middleware import AgentSafetyMiddleware
    app.add_middleware(AgentSafetyMiddleware)

Usage (Flask):
    from agent_safety_middleware import FlaskAgentSafety
    FlaskAgentSafety(app)

Usage (standalone):
    from agent_safety_middleware import SafetyGuard
    guard = SafetyGuard()
    result = guard.check("user input here")
"""

from agent_safety_middleware.guard import SafetyGuard, SafetyResult
from agent_safety_middleware.fastapi_middleware import AgentSafetyMiddleware
from agent_safety_middleware.flask_middleware import FlaskAgentSafety
from agent_safety_middleware.decorator import safe_endpoint

__all__ = [
    "SafetyGuard",
    "SafetyResult",
    "AgentSafetyMiddleware",
    "FlaskAgentSafety",
    "safe_endpoint",
]

__version__ = "0.1.0"
