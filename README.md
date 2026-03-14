# agent-safety-middleware

One-line safety middleware for AI agent APIs. Prompt injection scanning, cost budgets, decision audit trails.

## Install

```bash
pip install agent-safety-middleware
```

## FastAPI

```python
from fastapi import FastAPI
from agent_safety_middleware import AgentSafetyMiddleware

app = FastAPI()
app.add_middleware(AgentSafetyMiddleware)
```

## Flask

```python
from flask import Flask
from agent_safety_middleware import FlaskAgentSafety

app = Flask(__name__)
FlaskAgentSafety(app)
```

## Decorator

```python
from agent_safety_middleware import safe_endpoint

@app.post("/chat")
@safe_endpoint(injection_threshold=5, max_cost_per_request=0.50)
async def chat(prompt: str):
    ...
```

## Standalone

```python
from agent_safety_middleware import SafetyGuard

guard = SafetyGuard(injection_threshold=5, max_cost_per_session=10.00)
result = guard.check("user input here")
if not result.safe:
    print(f"Blocked: {result.blocked_reason}")
```

Automatically scans POST/PUT/PATCH request bodies for injection attacks across 75 patterns across 9 categories. Adds `X-Safety-*` response headers. Zero config required.
