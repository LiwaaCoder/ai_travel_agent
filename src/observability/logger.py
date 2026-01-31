"""
Structured logging for the AI Travel Agent.

Provides contextual logging with support for:
- Structured JSON output
- Request tracing
- Node-level debugging
- Performance metrics
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from functools import wraps
import os


# =============================================================================
# Log Formatters
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, "node"):
            log_data["node"] = record.node
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "city"):
            log_data["city"] = record.city
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "tokens"):
            log_data["tokens"] = record.tokens
            
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build prefix
        prefix = f"{color}{self.BOLD}[{record.levelname:>7}]{self.RESET}"
        
        # Add node context if available
        node_ctx = ""
        if hasattr(record, "node"):
            node_ctx = f" \033[90m[{record.node}]\033[0m"
        
        # Add latency if available
        latency_ctx = ""
        if hasattr(record, "latency_ms"):
            latency_ctx = f" \033[90m({record.latency_ms:.0f}ms)\033[0m"
        
        return f"{prefix} {timestamp}{node_ctx} {record.getMessage()}{latency_ctx}"


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the Travel Agent.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON format (for production)
        log_file: Optional file path for log output
    """
    root_logger = logging.getLogger("travel_agent")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (always JSON)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    root_logger.propagate = False


def get_logger(name: str = "travel_agent") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"travel_agent.{name}")


# =============================================================================
# Agent Logger
# =============================================================================

@dataclass
class NodeContext:
    """Context for a graph node execution."""
    node_name: str
    session_id: str
    city: Optional[str] = None
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def elapsed_ms(self) -> float:
        return (datetime.now().timestamp() - self.start_time) * 1000


class AgentLogger:
    """
    Structured logger for the Travel Agent with context tracking.
    
    Usage:
        logger = AgentLogger(session_id="abc123")
        
        with logger.node_context("retrieve_knowledge", city="Barcelona"):
            # ... node logic ...
            logger.info("Retrieved 5 documents")
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self._logger = get_logger("agent")
        self._current_node: Optional[NodeContext] = None
    
    @staticmethod
    def _generate_session_id() -> str:
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _log(self, level: int, message: str, **extra) -> None:
        """Log with context."""
        extra["session_id"] = self.session_id
        if self._current_node:
            extra["node"] = self._current_node.node_name
            extra["latency_ms"] = self._current_node.elapsed_ms()
            if self._current_node.city:
                extra["city"] = self._current_node.city
        
        # Create log record with extra fields
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "",
            0,
            message,
            (),
            None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        self._logger.handle(record)
    
    def debug(self, message: str, **extra) -> None:
        self._log(logging.DEBUG, message, **extra)
    
    def info(self, message: str, **extra) -> None:
        self._log(logging.INFO, message, **extra)
    
    def warning(self, message: str, **extra) -> None:
        self._log(logging.WARNING, message, **extra)
    
    def error(self, message: str, **extra) -> None:
        self._log(logging.ERROR, message, **extra)
    
    def node_context(self, node_name: str, **kwargs):
        """Context manager for node execution tracking."""
        return _NodeContextManager(self, node_name, **kwargs)
    
    def log_state_transition(self, from_node: str, to_node: str) -> None:
        """Log a state transition in the graph."""
        self.info(f"Transition: {from_node} → {to_node}")
    
    def log_retrieval(self, query: str, num_docs: int, sources: list[str]) -> None:
        """Log a RAG retrieval operation."""
        self.info(
            f"Retrieved {num_docs} documents",
            query=query[:50] + "..." if len(query) > 50 else query,
            sources=sources,
        )
    
    def log_llm_call(self, model: str, tokens: int, latency_ms: float) -> None:
        """Log an LLM API call."""
        self.info(
            f"LLM call: {model}",
            tokens=tokens,
            latency_ms=latency_ms,
        )
    
    def log_tool_call(self, tool_name: str, success: bool, latency_ms: float) -> None:
        """Log an external tool call."""
        status = "✓" if success else "✗"
        self.info(f"Tool {status}: {tool_name}", latency_ms=latency_ms)


class _NodeContextManager:
    """Context manager for node execution."""
    
    def __init__(self, logger: AgentLogger, node_name: str, **kwargs):
        self.logger = logger
        self.node_name = node_name
        self.kwargs = kwargs
        self._previous_node: Optional[NodeContext] = None
    
    def __enter__(self):
        self._previous_node = self.logger._current_node
        self.logger._current_node = NodeContext(
            node_name=self.node_name,
            session_id=self.logger.session_id,
            city=self.kwargs.get("city"),
        )
        self.logger.debug(f"Entering node: {self.node_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Node failed: {exc_val}")
        else:
            self.logger.debug(f"Completed node: {self.node_name}")
        self.logger._current_node = self._previous_node
        return False


# =============================================================================
# Initialize default logging
# =============================================================================

# Auto-setup based on environment
_log_level = os.getenv("LOG_LEVEL", "INFO")
_json_output = os.getenv("LOG_FORMAT", "").lower() == "json"
setup_logging(level=_log_level, json_output=_json_output)
