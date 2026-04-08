"""Centralised logging configuration for ZeroAutoCL."""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger.

    Each logger is configured with a StreamHandler (stdout) and an optional
    FileHandler.  Calling this function multiple times with the same *name*
    returns the same logger instance without adding duplicate handlers.

    Args:
        name: Logger name (typically the module's ``__name__``).
        level: Logging level, e.g. ``logging.DEBUG``.
        log_file: Optional path to write log output to disk.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers more than once when the module is re-imported.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Do not propagate to the root logger to avoid duplicate output.
    logger.propagate = False

    return logger
