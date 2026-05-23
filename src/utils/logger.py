"""
Utility functions for logging across the platform.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "healthcare", log_dir: str = "logs") -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler
    try:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        ))
        logger.addHandler(file_handler)
    except Exception:
        pass  # File logging is optional

    return logger


# Default logger
logger = setup_logger()
