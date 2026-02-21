"""Core utilities."""

import logging
import shutil
import subprocess
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from verification_ocr.core.settings.app_settings import LoggingSettings

logger = logging.getLogger(__name__)

# Handler names used to identify handlers and avoid duplicates
APP_STREAM_HANDLER_NAME = "vocr_app_stream_handler"
APP_FILE_HANDLER_NAME = "vocr_app_file_handler"


class HealthCheckFilter(logging.Filter):
    """Filter to exclude health check requests from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out health check log entries.

        Args:
            record (logging.LogRecord): Log record to check.

            bool: False to exclude the record, True to include it.
        """
        message = record.getMessage()
        # Filter out GET /health requests
        if "/health" in message and "GET" in message:
            return False
        return True


def get_tesseract_version() -> str | None:
    """
    Get Tesseract version string.

        str | None: Version string if tesseract is available, None otherwise.
    """
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return None

    try:
        result = subprocess.run(
            args=[tesseract_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # First line contains version info
        version_line = result.stdout.split("\n")[0] if result.stdout else None
        if version_line:
            return version_line.strip()
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        logger.warning(f"Failed to get tesseract version: {e}")
        return None


def _get_handler_by_name(root_logger: logging.Logger, name: str) -> logging.Handler | None:
    """
    Get a handler by name from a logger.

    Args:
        root_logger (logging.Logger): Logger to search.
        name (str): Handler name to find.

        logging.Handler | None: Handler if found, None otherwise.
    """
    for handler in root_logger.handlers:
        if getattr(handler, "name", None) == name:
            return handler
    return None


def _remove_handler_by_name(root_logger: logging.Logger, name: str) -> None:
    """
    Remove a handler by name from a logger.

    Args:
        root_logger (logging.Logger): Logger to remove from.
        name (str): Handler name to remove.

    """
    handler = _get_handler_by_name(root_logger=root_logger, name=name)
    if handler:
        root_logger.removeHandler(handler)
        handler.close()


def _get_min_level(root_level: str, loggers: dict[str, str]) -> int:
    """
    Get the minimum log level from root and all custom loggers.

    Args:
        root_level (str): The root logger level string.
        loggers (dict[str, str]): Dict of logger name to level string.

        int: The minimum numeric log level.
    """
    levels: list[int] = [logging.getLevelName(root_level.upper())]
    for level in loggers.values():
        levels.append(logging.getLevelName(level.upper()))
    return min(levels)


def setup_logging(settings: LoggingSettings) -> None:
    """
    Setup logging configuration for the application.

    Args:
        settings (LoggingSettings): Logging settings to configure logging.

    """
    root_logger = logging.getLogger()

    # Calculate minimum level across root and all custom loggers
    min_level = _get_min_level(
        root_level=settings.log_level,
        loggers=settings.loggers,
    )

    # Set the root logger level
    root_logger.setLevel(settings.log_level)

    # Remove any existing app handlers
    _remove_handler_by_name(root_logger=root_logger, name=APP_STREAM_HANDLER_NAME)
    _remove_handler_by_name(root_logger=root_logger, name=APP_FILE_HANDLER_NAME)

    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt=settings.date_format,
    )

    # Create and add the appropriate handler
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if settings.rotate_logs:
            handler: logging.Handler = TimedRotatingFileHandler(
                filename=log_path,
                when="midnight",
                encoding="utf-8",
            )
        else:
            handler = logging.FileHandler(log_path)

        handler.set_name(APP_FILE_HANDLER_NAME)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(APP_STREAM_HANDLER_NAME)

    handler.setFormatter(formatter)
    handler.setLevel(min_level)
    root_logger.addHandler(handler)

    # Reset common loggers to NOTSET so they inherit from root
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging.getLogger(logger_name).setLevel(logging.NOTSET)

    # Configure individual logger levels from settings
    for logger_name, level in settings.loggers.items():
        logging.getLogger(logger_name).setLevel(level)

    # Add filter to suppress health check logs from uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())
