"""Tests for core utilities."""

import logging
import subprocess
from unittest.mock import MagicMock, patch

from verification_ocr.core.utils import (
    HealthCheckFilter,
    get_tesseract_version,
    setup_logging,
)


class TestGetTesseractVersion:
    """Tests for get_tesseract_version function."""

    def test_tesseract_available(self, mock_tesseract_available: MagicMock) -> None:
        """
        Test when tesseract is available.

        Args:
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        result = get_tesseract_version()
        assert result == "tesseract 5.0.0"

    def test_tesseract_not_in_path(self, mock_tesseract_unavailable: MagicMock) -> None:
        """
        Test when tesseract is not in PATH.

        Args:
            mock_tesseract_unavailable (MagicMock): Mock for tesseract unavailability.

        Returns:
            None
        """
        result = get_tesseract_version()
        assert result is None

    def test_tesseract_empty_output(self) -> None:
        """
        Test when tesseract returns empty output.

        Returns:
            None
        """
        with patch(
            "verification_ocr.core.utils.shutil.which",
            return_value="/usr/bin/tesseract",
        ):
            with patch("verification_ocr.core.utils.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="",
                    stderr="",
                    returncode=0,
                )
                result = get_tesseract_version()
                assert result is None

    def test_tesseract_timeout(self) -> None:
        """
        Test when tesseract command times out.

        Returns:
            None
        """
        with patch(
            "verification_ocr.core.utils.shutil.which",
            return_value="/usr/bin/tesseract",
        ):
            with patch(
                "verification_ocr.core.utils.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=10),
            ):
                result = get_tesseract_version()
                assert result is None

    def test_tesseract_subprocess_error(self) -> None:
        """
        Test when subprocess raises an error.

        Returns:
            None
        """
        with patch(
            "verification_ocr.core.utils.shutil.which",
            return_value="/usr/bin/tesseract",
        ):
            with patch(
                "verification_ocr.core.utils.subprocess.run",
                side_effect=subprocess.SubprocessError("Test error"),
            ):
                result = get_tesseract_version()
                assert result is None

    def test_tesseract_os_error(self) -> None:
        """
        Test when OSError is raised.

        Returns:
            None
        """
        with patch(
            "verification_ocr.core.utils.shutil.which",
            return_value="/usr/bin/tesseract",
        ):
            with patch(
                "verification_ocr.core.utils.subprocess.run",
                side_effect=OSError("Permission denied"),
            ):
                result = get_tesseract_version()
                assert result is None


class TestHealthCheckFilter:
    """Tests for HealthCheckFilter."""

    def test_filters_health_check_get_request(self) -> None:
        """
        Test that GET /health requests are filtered out.

        Returns:
            None
        """
        filter_instance = HealthCheckFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='127.0.0.1:8000 - "GET /health HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is False

    def test_allows_other_requests(self) -> None:
        """
        Test that non-health check requests are allowed.

        Returns:
            None
        """
        filter_instance = HealthCheckFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='127.0.0.1:8000 - "POST /verify HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True

    def test_allows_health_post_request(self) -> None:
        """
        Test that POST to /health is allowed (only GET is filtered).

        Returns:
            None
        """
        filter_instance = HealthCheckFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='127.0.0.1:8000 - "POST /health HTTP/1.1" 200',
            args=(),
            exc_info=None,
        )
        assert filter_instance.filter(record) is True


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """
        Test setup_logging with default settings.

        Returns:
            None
        """
        from verification_ocr.core.settings.app_settings import LoggingSettings

        settings = LoggingSettings()
        setup_logging(settings=settings)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_custom_level(self) -> None:
        """
        Test setup_logging with custom level.

        Returns:
            None
        """
        from verification_ocr.core.settings.app_settings import LoggingSettings

        settings = LoggingSettings(log_level="DEBUG")
        setup_logging(settings=settings)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_custom_loggers(self) -> None:
        """
        Test setup_logging with custom logger levels.

        Returns:
            None
        """
        from verification_ocr.core.settings.app_settings import LoggingSettings

        settings = LoggingSettings(
            log_level="INFO",
            loggers={"test_logger": "WARNING"},
        )
        setup_logging(settings=settings)

        test_logger = logging.getLogger("test_logger")
        assert test_logger.level == logging.WARNING

    def test_setup_logging_with_file(self, tmp_path) -> None:
        """
        Test setup_logging with log file.

        Args:
            tmp_path: Pytest fixture for temporary directory.

        Returns:
            None
        """
        from verification_ocr.core.settings.app_settings import LoggingSettings

        log_file = tmp_path / "test.log"
        settings = LoggingSettings(log_file=str(log_file))
        setup_logging(settings=settings)

        # Verify file handler was added
        root_logger = logging.getLogger()
        file_handlers = [
            h
            for h in root_logger.handlers
            if hasattr(h, "name") and h.name == "vocr_app_file_handler"
        ]
        assert len(file_handlers) == 1

    def test_setup_logging_health_check_filter(self) -> None:
        """
        Test that health check filter is added to uvicorn.access logger.

        Returns:
            None
        """
        from verification_ocr.core.settings.app_settings import LoggingSettings
        from verification_ocr.core.utils import HealthCheckFilter

        settings = LoggingSettings()
        setup_logging(settings=settings)

        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        health_filters = [
            f for f in uvicorn_access_logger.filters if isinstance(f, HealthCheckFilter)
        ]
        assert len(health_filters) >= 1

    def test_setup_logging_with_rotating_file(self, tmp_path) -> None:
        """
        Test setup_logging with rotating log file.

        Args:
            tmp_path: Pytest fixture for temporary directory.

        Returns:
            None
        """
        from logging.handlers import TimedRotatingFileHandler

        from verification_ocr.core.settings.app_settings import LoggingSettings

        log_file = tmp_path / "test_rotating.log"
        settings = LoggingSettings(log_file=str(log_file), rotate_logs=True)
        setup_logging(settings=settings)

        # Verify TimedRotatingFileHandler was added
        root_logger = logging.getLogger()
        rotating_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, TimedRotatingFileHandler)
            and hasattr(h, "name")
            and h.name == "vocr_app_file_handler"
        ]
        assert len(rotating_handlers) == 1
