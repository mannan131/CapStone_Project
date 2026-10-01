"""Structured logging via structlog."""

from __future__ import annotations

import logging

import structlog

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def get_logger(name: str = "churn-engine", level: str = "info"):
    lvl = _LEVELS.get(str(level).lower(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
    )
    return structlog.get_logger(name)
