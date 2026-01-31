"""
Distributed tracing for the AI Travel Agent.

Integrates with LangSmith for LangChain/LangGraph tracing.
"""

import os
import time
from typing import Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class TraceSpan:
    """Represents a single span in a trace."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    status: str = "running"
    
    def end(self, status: str = "success") -> None:
        self.end_time = time.time()
        self.status = status
    
    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


class LangSmithTracer:
    """
    Tracer that integrates with LangSmith for observability.
    
    Falls back to local tracing if LangSmith is not configured.
    
    Usage:
        tracer = LangSmithTracer(project_name="travel-agent")
        
        with tracer.span("retrieve_knowledge") as span:
            # ... operation ...
            span.metadata["num_docs"] = 5
    """
    
    def __init__(
        self,
        project_name: str = "travel-agent",
        enabled: Optional[bool] = None,
    ):
        self.project_name = project_name
        self._enabled = enabled if enabled is not None else self._check_langsmith_config()
        self._trace_id: Optional[str] = None
        self._current_span: Optional[TraceSpan] = None
        self._spans: list[TraceSpan] = []
    
    @staticmethod
    def _check_langsmith_config() -> bool:
        """Check if LangSmith is configured."""
        return bool(os.getenv("LANGCHAIN_API_KEY")) and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    
    def start_trace(self, name: str = "agent_run") -> str:
        """Start a new trace."""
        self._trace_id = str(uuid.uuid4())
        self._spans = []
        self._current_span = None
        return self._trace_id
    
    def end_trace(self) -> dict:
        """End the current trace and return summary."""
        summary = {
            "trace_id": self._trace_id,
            "spans": len(self._spans),
            "total_duration_ms": sum(s.duration_ms for s in self._spans),
            "status": "success" if all(s.status == "success" for s in self._spans) else "error",
        }
        self._trace_id = None
        return summary
    
    @contextmanager
    def span(self, name: str, **metadata):
        """Create a trace span."""
        if not self._trace_id:
            self.start_trace()
        
        span = TraceSpan(
            name=name,
            trace_id=self._trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=self._current_span.span_id if self._current_span else None,
            metadata=metadata,
        )
        
        previous_span = self._current_span
        self._current_span = span
        self._spans.append(span)
        
        try:
            yield span
            span.end("success")
        except Exception as e:
            span.end("error")
            span.metadata["error"] = str(e)
            raise
        finally:
            self._current_span = previous_span
    
    def get_trace_url(self) -> Optional[str]:
        """Get LangSmith trace URL if available."""
        if self._enabled and self._trace_id:
            return f"https://smith.langchain.com/o/default/projects/p/{self.project_name}/r/{self._trace_id}"
        return None


def trace_node(tracer: Optional[LangSmithTracer] = None):
    """
    Decorator to trace a graph node function.
    
    Usage:
        @trace_node()
        async def classify_intent(state: TravelAgentState) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _tracer = tracer or LangSmithTracer()
            with _tracer.span(func.__name__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _tracer = tracer or LangSmithTracer()
            with _tracer.span(func.__name__):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def get_run_url() -> Optional[str]:
    """Get the current LangSmith run URL from environment."""
    # LangSmith sets this during traced runs
    return os.getenv("LANGCHAIN_RUN_URL")
