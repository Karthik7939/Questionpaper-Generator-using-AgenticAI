"""
services/logger.py

Centralized logging service for the Agentic Question Paper Generator.

Features:
  - Console output with color-coded log levels
  - Rotating file handler (logs/application.log)
  - Per-agent execution timing context manager
  - Token usage logging helper
  - Single shared logger instance per module name
"""

import logging
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Generator

from app.config import settings

# ---------------------------------------------------------------------------
# ANSI color codes for console output
# ---------------------------------------------------------------------------
_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Custom formatter that injects ANSI color codes into console output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Internal registry — one logger per name, never duplicated
# ---------------------------------------------------------------------------
_loggers: dict[str, logging.Logger] = {}
_root_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once (file + console handlers)."""
    global _root_configured
    if _root_configured:
        return

    # Ensure the logs directory exists before creating the file handler
    settings.paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("qpgen")
    root_logger.setLevel(getattr(logging, settings.log.LOG_LEVEL, logging.INFO))

    # ---- Console handler (colored) ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = _ColorFormatter(
        fmt=settings.log.LOG_FORMAT,
        datefmt=settings.log.DATE_FORMAT,
    )
    console_handler.setFormatter(console_formatter)

    # ---- File handler (rotating, plain text) ----
    file_handler = RotatingFileHandler(
        filename=settings.log.LOG_FILE,
        maxBytes=settings.log.MAX_BYTES,
        backupCount=settings.log.BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt=settings.log.LOG_FORMAT,
        datefmt=settings.log.DATE_FORMAT,
    )
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    _root_configured = True


def setup_logger(name: str) -> logging.Logger:
    """
    Get or create a named logger under the 'qpgen' hierarchy.

    Usage:
        logger = setup_logger(__name__)

    Args:
        name: Module name (pass __name__ from calling module).

    Returns:
        A configured Logger instance.
    """
    _configure_root_logger()

    if name not in _loggers:
        logger = logging.getLogger(f"qpgen.{name}")
        _loggers[name] = logger

    return _loggers[name]


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------
@contextmanager
def log_execution_time(
    logger: logging.Logger,
    agent_name: str,
) -> Generator[None, None, None]:
    """
    Context manager that logs agent start, completion, and elapsed time.

    Usage:
        with log_execution_time(logger, "SyllabusAgent"):
            result = agent.run(state)

    Args:
        logger:     The module logger instance.
        agent_name: Human-readable name of the agent or step being timed.
    """
    logger.info(f"▶ [{agent_name}] Execution started")
    start_time = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"✖ [{agent_name}] Failed after {elapsed:.2f}s — {type(exc).__name__}: {exc}"
        )
        raise
    else:
        elapsed = time.perf_counter() - start_time
        logger.info(f"✔ [{agent_name}] Completed in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Token usage logger
# ---------------------------------------------------------------------------
def log_token_usage(
    logger: logging.Logger,
    agent_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """
    Log LLM token consumption for a given agent call.

    Args:
        logger:            The module logger instance.
        agent_name:        Name of the calling agent.
        prompt_tokens:     Tokens used in the prompt.
        completion_tokens: Tokens used in the completion.
    """
    total = prompt_tokens + completion_tokens
    logger.info(
        f"[{agent_name}] Token usage — "
        f"prompt: {prompt_tokens} | "
        f"completion: {completion_tokens} | "
        f"total: {total}"
    )
