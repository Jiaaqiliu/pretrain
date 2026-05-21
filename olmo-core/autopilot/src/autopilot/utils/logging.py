"""Structured logging for AutoPilot with rich console output."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(f"autopilot.{name}")
    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    root = logging.getLogger("autopilot")
    root.setLevel(level)

    if not root.handlers:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        root.addHandler(file_handler)
