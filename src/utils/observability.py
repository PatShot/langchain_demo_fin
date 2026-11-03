"""LangSmith tracing.

LangChain and LangGraph send traces to LangSmith automatically when the right
environment variables are set, so there is no extra code to write. This helper
just reports whether that is switched on.
"""

import os

# The `.env` file is loaded by `src.models.llm_models` when it is imported,
# which always happens before this function is called.


def tracing_status() -> str:
    """Describe whether runs are being traced to LangSmith."""
    if os.getenv("LANGSMITH_TRACING", "").lower() not in {"true", "1", "yes"}:
        return "LangSmith tracing is off (set LANGSMITH_TRACING=true to turn it on)."
    if not os.getenv("LANGSMITH_API_KEY"):
        return "LangSmith tracing is on, but LANGSMITH_API_KEY is missing - no traces will be sent."
    project = os.getenv("LANGSMITH_PROJECT", "default")
    return f"LangSmith tracing is on, sending runs to project '{project}'."
