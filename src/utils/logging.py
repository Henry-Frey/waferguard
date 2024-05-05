"""Structured logging.

Console output is always enabled (pretty-printed via ConsoleRenderer).
When the LOG_FILE environment variable is set, a second JSON handler writes
each log record as a single-line JSON object to that path — Filebeat picks
this file up and ships it to the ELK stack.

Example (container use-case):
    LOG_FILE=/app/logs/waferguard.jsonl uvicorn src.serving.app:app ...
"""

from __future__ import annotations

import logging as stdlib_logging
import os

import structlog


def setup_logging(level: str = "INFO") -> None:
    log_file = os.getenv("LOG_FILE")

    # Processors shared by both the console and JSON handlers.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Bridge structlog → stdlib so we can attach multiple handlers.
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = stdlib_logging.getLogger()
    root.setLevel(stdlib_logging.getLevelName(level))
    root.handlers.clear()

    # ── Console handler (always on) ──────────────────────────────────────
    console_handler = stdlib_logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(),
            ]
        )
    )
    root.addHandler(console_handler)

    # ── JSON file handler (ELK integration, opt-in via LOG_FILE) ─────────
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = stdlib_logging.FileHandler(log_file)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ]
            )
        )
        root.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
