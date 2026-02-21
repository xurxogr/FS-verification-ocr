"""Tests for application settings."""

import os
from unittest.mock import patch

from verification_ocr.core.settings.app_settings import (
    APIServerSettings,
    AppSettings,
    LoggingSettings,
    OCRSettings,
    VerificationSettings,
    WarSettings,
)


class TestAPIServerSettings:
    """Tests for APIServerSettings."""

    def test_default_host(self) -> None:
        """
        Test default host value.

        """
        settings = APIServerSettings()
        assert settings.host == "0.0.0.0"

    def test_default_port(self) -> None:
        """
        Test default port value.

        """
        settings = APIServerSettings()
        assert settings.port == 8000

    def test_default_workers(self) -> None:
        """
        Test default workers value.

        """
        settings = APIServerSettings()
        assert settings.workers == 1

    def test_default_cors_allow_origins(self) -> None:
        """
        Test default CORS allow origins.

        """
        settings = APIServerSettings()
        assert settings.cors_allow_origins == ["*"]

    def test_custom_values(self) -> None:
        """
        Test custom values.

        """
        settings = APIServerSettings(
            host="127.0.0.1",
            port=9000,
            workers=4,
            cors_allow_origins=["http://localhost:3000"],
        )
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.workers == 4
        assert settings.cors_allow_origins == ["http://localhost:3000"]


class TestVerificationSettings:
    """Tests for VerificationSettings."""

    def test_default_max_ingame_time_diff(self) -> None:
        """
        Test default max_ingame_time_diff.

        """
        settings = VerificationSettings()
        assert settings.max_ingame_time_diff == 25

    def test_custom_max_ingame_time_diff(self) -> None:
        """
        Test custom max_ingame_time_diff.

        """
        settings = VerificationSettings(max_ingame_time_diff=60)
        assert settings.max_ingame_time_diff == 60


class TestOCRSettings:
    """Tests for OCRSettings."""

    def test_default_tesseract_cmd(self) -> None:
        """
        Test default tesseract_cmd is None.

        """
        settings = OCRSettings()
        assert settings.tesseract_cmd is None

    def test_default_language(self) -> None:
        """
        Test default language includes multiple languages.

        """
        settings = OCRSettings()
        assert "eng" in settings.language
        assert "fra" in settings.language

    def test_default_colonial_icon_path(self) -> None:
        """
        Test default colonial_icon_path.

        """
        settings = OCRSettings()
        assert settings.colonial_icon_path == "data/colonial_icon.png"

    def test_default_scale_factor(self) -> None:
        """
        Test default scale_factor.

        """
        settings = OCRSettings()
        assert settings.scale_factor == 4

    def test_default_clahe_settings(self) -> None:
        """
        Test default CLAHE settings.

        """
        settings = OCRSettings()
        assert settings.clahe_clip_limit == 2.0
        assert settings.clahe_grid_size == 8

    def test_default_base_dimensions(self) -> None:
        """
        Test default base dimensions for scaling.

        """
        settings = OCRSettings()
        assert settings.base_height == 2160
        assert settings.base_box_width == 84
        assert settings.base_box_height == 64

    def test_default_debug_settings(self) -> None:
        """
        Test default debug settings.

        """
        settings = OCRSettings()
        assert settings.debug_mode is False
        assert settings.debug_output_dir == "screenshots"

    def test_custom_tesseract_cmd(self) -> None:
        """
        Test custom tesseract_cmd.

        """
        settings = OCRSettings(tesseract_cmd="/custom/tesseract")
        assert settings.tesseract_cmd == "/custom/tesseract"

    def test_custom_language(self) -> None:
        """
        Test custom language.

        """
        settings = OCRSettings(language="spa")
        assert settings.language == "spa"

    def test_custom_colonial_icon_path(self) -> None:
        """
        Test custom colonial_icon_path.

        """
        settings = OCRSettings(colonial_icon_path="/path/to/icon.png")
        assert settings.colonial_icon_path == "/path/to/icon.png"


class TestWarSettings:
    """Tests for WarSettings."""

    def test_default_number(self) -> None:
        """
        Test default war number is None.

        """
        settings = WarSettings()
        assert settings.number is None

    def test_default_start_time(self) -> None:
        """
        Test default start_time is None.

        """
        settings = WarSettings()
        assert settings.start_time is None

    def test_custom_number(self) -> None:
        """
        Test custom war number.

        """
        settings = WarSettings(number=132)
        assert settings.number == 132

    def test_custom_start_time(self) -> None:
        """
        Test custom start_time.

        """
        settings = WarSettings(start_time=1770663602746)
        assert settings.start_time == 1770663602746


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_default_log_level(self) -> None:
        """
        Test default log level.

        """
        settings = LoggingSettings()
        assert settings.log_level == "INFO"

    def test_default_log_format(self) -> None:
        """
        Test default log format.

        """
        settings = LoggingSettings()
        assert "%(asctime)s" in settings.log_format
        assert "%(levelname)s" in settings.log_format

    def test_default_date_format(self) -> None:
        """
        Test default date format.

        """
        settings = LoggingSettings()
        assert settings.date_format == "%Y-%m-%d %H:%M:%S"

    def test_default_loggers(self) -> None:
        """
        Test default loggers dict.

        """
        settings = LoggingSettings()
        assert settings.loggers == {}

    def test_default_rotate_logs(self) -> None:
        """
        Test default rotate_logs.

        """
        settings = LoggingSettings()
        assert settings.rotate_logs is False

    def test_default_log_file(self) -> None:
        """
        Test default log_file.

        """
        settings = LoggingSettings()
        assert settings.log_file is None

    def test_custom_log_level(self) -> None:
        """
        Test custom log level.

        """
        settings = LoggingSettings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_custom_loggers(self) -> None:
        """
        Test custom loggers.

        """
        settings = LoggingSettings(loggers={"uvicorn": "WARNING"})
        assert settings.loggers == {"uvicorn": "WARNING"}


class TestAppSettings:
    """Tests for AppSettings."""

    def test_default_api_server(self) -> None:
        """
        Test default api_server settings.

        """
        settings = AppSettings()
        assert isinstance(settings.api_server, APIServerSettings)

    def test_default_ocr(self) -> None:
        """
        Test default ocr settings.

        """
        settings = AppSettings()
        assert isinstance(settings.ocr, OCRSettings)

    def test_default_logging(self) -> None:
        """
        Test default logging settings.

        """
        settings = AppSettings()
        assert isinstance(settings.logging, LoggingSettings)

    def test_default_war(self) -> None:
        """
        Test default war settings.

        """
        settings = AppSettings()
        assert isinstance(settings.war, WarSettings)

    def test_default_verification(self) -> None:
        """
        Test default verification settings.

        """
        settings = AppSettings()
        assert isinstance(settings.verification, VerificationSettings)

    def test_env_prefix(self) -> None:
        """
        Test that environment variables with VOCR_ prefix are loaded.

        """
        with patch.dict(
            os.environ,
            {"VOCR_API_SERVER__PORT": "9999"},
            clear=False,
        ):
            settings = AppSettings()
            assert settings.api_server.port == 9999

    def test_env_nested_delimiter(self) -> None:
        """
        Test that nested settings use __ delimiter.

        """
        with patch.dict(
            os.environ,
            {"VOCR_OCR__LANGUAGE": "fra"},
            clear=False,
        ):
            settings = AppSettings()
            assert settings.ocr.language == "fra"

    def test_custom_nested_settings(self) -> None:
        """
        Test creating settings with custom nested values.

        """
        settings = AppSettings(
            api_server=APIServerSettings(port=5000),
            ocr=OCRSettings(language="deu"),
        )
        assert settings.api_server.port == 5000
        assert settings.ocr.language == "deu"
