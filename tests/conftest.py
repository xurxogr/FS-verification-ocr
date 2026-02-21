"""Pytest configuration and fixtures."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from verification_ocr.api.server import app
from verification_ocr.core.settings import AppSettings, reload_settings
from verification_ocr.core.settings.app_settings import (
    APIServerSettings,
    LoggingSettings,
    OCRSettings,
)
from verification_ocr.services import get_war_service


@pytest.fixture
def mock_settings() -> AppSettings:
    """
    Create mock application settings for testing.

        AppSettings: Mock settings instance.
    """
    return AppSettings(
        api_server=APIServerSettings(
            host="127.0.0.1",
            port=8000,
            workers=1,
            cors_allow_origins=["http://localhost:3000"],
            max_upload_size=50 * 1024 * 1024,
            rate_limit="100/minute",
        ),
        ocr=OCRSettings(
            tesseract_cmd=None,
            language="eng",
            colonial_icon_path=None,
            wardens_icon_path=None,
            debug_mode=False,
        ),
        logging=LoggingSettings(
            log_level="DEBUG",
            log_format="%(message)s",
        ),
    )


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Create sample image bytes for testing.

        bytes: PNG image bytes.
    """
    # Create a simple 10x10 white image with a black pixel
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    img[5, 5] = [0, 0, 0]  # Black pixel in center
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


@pytest.fixture
def sample_image_with_text_bytes() -> bytes:
    """
    Create sample image bytes with text for OCR testing.

        bytes: PNG image bytes with text.
    """
    # Create a larger image for text
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(
        img=img,
        text="TEST",
        org=(50, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0, 0, 0),
        thickness=2,
    )
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


@pytest.fixture
def invalid_image_bytes() -> bytes:
    """
    Create invalid image bytes for testing error handling.

        bytes: Invalid image data.
    """
    return b"not a valid image"


@pytest.fixture
def integration_settings() -> AppSettings:
    """
    Create settings for integration tests with real faction icons.

        AppSettings: Settings instance with real icon paths.
    """
    return AppSettings(
        api_server=APIServerSettings(
            host="127.0.0.1",
            port=8000,
            workers=1,
            cors_allow_origins=["http://localhost:3000"],
            max_upload_size=50 * 1024 * 1024,
            rate_limit="100/minute",
        ),
        ocr=OCRSettings(
            tesseract_cmd=None,
            language="eng",
            colonial_icon_path="data/colonial_icon.png",
            wardens_icon_path="data/wardens_icon.png",
            debug_mode=False,
        ),
        logging=LoggingSettings(
            log_level="DEBUG",
            log_format="%(message)s",
        ),
    )


@pytest.fixture
def colonial_image_bytes() -> bytes:
    """
    Load real colonial user screenshot for integration testing.

        bytes: PNG image bytes from fixtures.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "colonial.png",
    )
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture
def warden_image_bytes() -> bytes:
    """
    Load real warden user screenshot for integration testing.

        bytes: PNG image bytes from fixtures.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "warden.png",
    )
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture
def stockpile_image_bytes() -> bytes:
    """
    Load real stockpile screenshot for integration testing.

        bytes: PNG image bytes from fixtures.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "stockpile.png",
    )
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """
    Reset settings cache before each test.

    """
    reload_settings()


@pytest.fixture(autouse=True)
def reset_war_state() -> None:
    """
    Reset war state before each test.

    """
    war_service = get_war_service()
    war_service.initialize(war_number=None, start_time=None)


@pytest.fixture
def mock_tesseract_available() -> Generator[MagicMock, None, None]:
    """
    Mock tesseract as available.

    Yields:
        MagicMock: The mock object.
    """
    with patch(
        "verification_ocr.core.utils.shutil.which",
        return_value="/usr/bin/tesseract",
    ) as mock_which:
        with patch("verification_ocr.core.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="tesseract 5.0.0\n leptonica-1.80.0",
                stderr="",
                returncode=0,
            )
            yield mock_which


@pytest.fixture
def mock_tesseract_unavailable() -> Generator[MagicMock, None, None]:
    """
    Mock tesseract as unavailable.

    Yields:
        MagicMock: The mock object.
    """
    with patch(
        "verification_ocr.core.utils.shutil.which",
        return_value=None,
    ) as mock_which:
        yield mock_which


@pytest.fixture
def test_client(mock_tesseract_available: MagicMock) -> TestClient:
    """
    Create a test client for the FastAPI application.

    Args:
        mock_tesseract_available (MagicMock): Mock for tesseract availability.

        TestClient: FastAPI test client.
    """
    return TestClient(app)
