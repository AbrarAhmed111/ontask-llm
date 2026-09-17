"""
Logging Configuration.
Sets up standardized console logging format for the application.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configures structured console logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Log messages use emoji as visual markers (see gateway.py, summary_service.py).
    # On Windows, stdout is often attached to a legacy codepage (e.g. cp1252) rather
    # than UTF-8, which makes those emoji raise UnicodeEncodeError mid-request. Force
    # UTF-8 on the stream so logging never crashes a request over a cosmetic choice.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Silence overly verbose external libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
