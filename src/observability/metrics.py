"""
Metrics collection for the AI Travel Agent.

Tracks performance metrics for monitoring and optimization.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import wraps
from collections import defaultdict
from datetime import datetime
import threading


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for the Travel Agent.
    
    Tracks:
    - Node latencies
    - LLM token usage
    - Retrieval performance
    - Tool success rates
    
    Usage:
        metrics = MetricsCollector()
        
        with metrics.timer("retrieve_knowledge"):
            # ... operation ...
        
        metrics.increment("documents_retrieved", 5)
        metrics.summary()
    """
    
    def __init__(self):
        self._metrics: list[MetricPoint] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._timers: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, **tags) -> None:
        """Record a metric value."""
        with self._lock:
            self._metrics.append(MetricPoint(name=name, value=value, tags=tags))
    
    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value
    
    def timer(self, name: str):
        """Context manager for timing operations."""
        return _TimerContext(self, name)
    
    def record_latency(self, name: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._timers[name].append(latency_ms)
        self.record(f"{name}_latency_ms", latency_ms)
    
    def summary(self) -> dict:
        """Get a summary of collected metrics."""
        with self._lock:
            timer_stats = {}
            for name, values in self._timers.items():
                if values:
                    timer_stats[name] = {
                        "count": len(values),
                        "mean_ms": sum(values) / len(values),
                        "min_ms": min(values),
                        "max_ms": max(values),
                        "total_ms": sum(values),
                    }
            
            return {
                "counters": dict(self._counters),
                "timers": timer_stats,
                "total_metrics": len(self._metrics),
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._timers.clear()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in self._counters.items():
            lines.append(f"travel_agent_{name}_total {value}")
        
        # Timer summaries
        for name, values in self._timers.items():
            if values:
                lines.append(f"travel_agent_{name}_count {len(values)}")
                lines.append(f"travel_agent_{name}_sum {sum(values)}")
        
        return "\n".join(lines)


class _TimerContext:
    """Context manager for timing."""
    
    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.collector.record_latency(self.name, elapsed_ms)
        return False


# =============================================================================
# Decorators
# =============================================================================

def track_latency(metrics: Optional[MetricsCollector] = None, name: Optional[str] = None):
    """
    Decorator to track function latency.
    
    Usage:
        @track_latency(name="classify_intent")
        async def classify_intent(state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _metrics = metrics or MetricsCollector()
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                _metrics.record_latency(metric_name, elapsed_ms)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _metrics = metrics or MetricsCollector()
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                _metrics.record_latency(metric_name, elapsed_ms)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_tokens(metrics: Optional[MetricsCollector] = None):
    """
    Decorator to track LLM token usage.
    
    Expects the decorated function to return a response with usage info.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _metrics = metrics or MetricsCollector()
            response = await func(*args, **kwargs)
            
            # Try to extract token usage from response
            if hasattr(response, "usage"):
                _metrics.increment("prompt_tokens", response.usage.prompt_tokens)
                _metrics.increment("completion_tokens", response.usage.completion_tokens)
                _metrics.increment("total_tokens", response.usage.total_tokens)
            
            return response
        
        return wrapper
    
    return decorator


# =============================================================================
# Global metrics instance
# =============================================================================

_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
