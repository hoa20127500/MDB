"""
Structured JSON logging setup with request-ID support.

Usage
-----
Call ``configure_logging()`` once at application startup (before any log
messages are emitted).  In FastAPI middleware or route handlers, set the
current request ID with::

    from app.core.logging import request_id_var
    token = request_id_var.set("req-abc123")
    try:
        ...
    finally:
        request_id_var.reset(token)

Every log record emitted while the context variable is set will include a
``request_id`` field in the JSON output.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ── Context variable for request-scoped IDs ──────────────────────────────────

#: Set this context variable to a non-empty string at the start of each
#: request so that all log records emitted during that request carry the same
#: ``request_id`` field.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# ── JSON formatter ───────────────────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Each record includes at minimum:
    - ``timestamp``  — ISO-8601 UTC timestamp
    - ``level``      — log level name (e.g. ``"INFO"``)
    - ``logger``     — logger name
    - ``message``    — formatted log message
    - ``request_id`` — value of :data:`request_id_var` (empty string if unset)

    If the record carries exception information, a ``exc_info`` field with the
    formatted traceback is appended.

    Extra fields passed via ``extra={"key": value}`` in the logging call are
    merged into the top-level JSON object.
    """

    # Fields that are already handled explicitly and should not be duplicated
    # from the record's ``__dict__``.
    _SKIP_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Build the base payload.
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }

        # Merge any extra fields the caller passed.
        for key, value in record.__dict__.items():
            if key not in self._SKIP_ATTRS and not key.startswith("_"):
                payload[key] = value

        # Append exception traceback when present.
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc_info"] = record.exc_text

        # Append stack info when present.
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


# ── Public setup function ────────────────────────────────────────────────────


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = sys.stdout,
) -> None:
    """Configure the root logger to emit structured JSON to *stream*.

    This function is idempotent: calling it multiple times replaces the
    existing handlers on the root logger rather than adding duplicates.

    Args:
        level: Minimum log level.  Accepts ``logging`` integer constants
            (e.g. ``logging.DEBUG``) or their string equivalents
            (e.g. ``"DEBUG"``).  Defaults to ``logging.INFO``.
        stream: Output stream.  Defaults to ``sys.stdout`` so that log lines
            are captured by container runtimes and log aggregators that read
            stdout.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Remove any handlers that were registered before this call.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Suppress overly verbose third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Convenience wrapper so callers don't need to import ``logging`` directly.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
