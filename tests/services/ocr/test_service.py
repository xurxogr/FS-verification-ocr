"""Tests for OCR service module."""

from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from verification_ocr.core.settings import get_settings
from verification_ocr.enums import Faction
from verification_ocr.models import Verification
from verification_ocr.services.ocr import OCRService
from verification_ocr.services.ocr.profile import Box
from verification_ocr.services.ocr.service import (
    ProfileData,
    ShardData,
    _get_current_ingame_time,
)


class TestOCRServiceInit:
    """Tests for OCRService initialization."""

    def test_init_with_settings(self) -> None:
        """Test service initialization with settings."""
        settings = get_settings()
        service = OCRService(settings)
        assert service.settings == settings

    def test_init_debug_mode(self) -> None:
        """Test service initialization with debug mode."""
        settings = get_settings()
        service = OCRService(settings, debug_mode=True)
        assert service.debug_mode is True

    def test_init_loads_faction_icons(self) -> None:
        """Test that faction icons are loaded during init."""
        settings = get_settings()
        service = OCRService(settings)
        # Icons may or may not be loaded depending on settings
        # Just verify the attributes exist
        assert hasattr(service, "colonial_icon")
        assert hasattr(service, "warden_icon")

    def test_init_sets_tesseract_cmd(self) -> None:
        """Test that tesseract_cmd is set from settings."""
        settings = get_settings()
        with patch.object(settings.ocr, "tesseract_cmd", "/custom/tesseract"):
            with patch(
                "verification_ocr.services.ocr.service.pytesseract.pytesseract"
            ) as mock_pytesseract:
                OCRService(settings)
                assert mock_pytesseract.tesseract_cmd == "/custom/tesseract"

    def test_init_creates_debug_dir(self) -> None:
        """Test that debug directory is created when debug_mode enabled in settings."""
        settings = get_settings()
        with patch.object(settings.ocr, "debug_mode", True):
            with patch.object(settings.ocr, "debug_output_dir", "/tmp/test_debug"):
                with patch("verification_ocr.services.ocr.service.os.makedirs") as mock_makedirs:
                    OCRService(settings)
                    mock_makedirs.assert_called_once_with(name="/tmp/test_debug", exist_ok=True)

    def test_load_icon_returns_none_for_empty_path(self) -> None:
        """Test that _load_icon returns None for empty path."""
        settings = get_settings()
        service = OCRService(settings)
        result = service._load_icon(path="", name="test")
        assert result is None

    def test_load_icon_returns_none_for_invalid_path(self) -> None:
        """Test that _load_icon returns None and logs warning for invalid path."""
        settings = get_settings()
        service = OCRService(settings)
        with patch("verification_ocr.services.ocr.service.logger") as mock_logger:
            result = service._load_icon(path="/nonexistent/icon.png", name="test")
            assert result is None
            mock_logger.warning.assert_called_once()


class TestValidateImage:
    """Tests for _validate_image method."""

    def test_raises_for_invalid_bytes(self) -> None:
        """Test that invalid bytes raise ValueError."""
        settings = get_settings()
        service = OCRService(settings)
        with pytest.raises(ValueError, match="Failed to decode"):
            service._validate_image(b"not an image")

    def test_raises_for_small_image(self) -> None:
        """Test that small images raise ValueError."""
        settings = get_settings()
        service = OCRService(settings)
        # Create a tiny image
        tiny = np.ones((50, 50, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", tiny)
        with pytest.raises(ValueError, match="Image too small"):
            service._validate_image(buffer.tobytes())

    def test_accepts_valid_image(self) -> None:
        """Test that valid images are accepted."""
        settings = get_settings()
        service = OCRService(settings)
        # Create a valid image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        result = service._validate_image(buffer.tobytes())
        assert result.shape == (500, 500, 3)


class TestDetectProfileBoxes:
    """Tests for detect_profile_boxes method."""

    def test_returns_none_for_blank_image(self) -> None:
        """Test that blank images return None."""
        settings = get_settings()
        service = OCRService(settings)
        blank = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        result = service.detect_profile_boxes(blank)
        assert result is None


class TestCropBox:
    """Tests for _crop_box method."""

    def test_crops_correct_region(self) -> None:
        """Test that correct region is cropped."""
        settings = get_settings()
        service = OCRService(settings)
        # Create image with known pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:40, 30:70] = [255, 0, 0]  # Blue region
        box: Box = (30, 20, 40, 20)
        crop = service._crop_box(img, box)
        assert crop.shape == (20, 40, 3)
        assert np.all(crop == [255, 0, 0])


class TestPreprocessForOCR:
    """Tests for _preprocess_for_ocr method."""

    def test_converts_to_grayscale(self) -> None:
        """Test that image is converted to grayscale."""
        settings = get_settings()
        service = OCRService(settings)
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = [255, 0, 0]  # Blue
        result = service._preprocess_for_ocr(color_img)
        assert len(result.shape) == 2  # Grayscale


class TestExtractText:
    """Tests for _extract_text method."""

    def test_extracts_text_with_mock(self) -> None:
        """Test text extraction with mocked pytesseract."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((50, 200, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="  TestText  \n",
        ):
            result = service._extract_text(img)
            assert result == "TestText"


class TestExtractLevel:
    """Tests for _extract_level method."""

    def test_extracts_numeric_level(self) -> None:
        """Test level extraction with mocked pytesseract."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((50, 100, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="42",
        ):
            result = service._extract_level(img)
            assert result == 42

    def test_returns_none_for_non_numeric(self) -> None:
        """Test level returns None for non-numeric text."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((50, 100, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="abc",
        ):
            result = service._extract_level(img)
            assert result is None


class TestExtractRegiment:
    """Tests for _extract_regiment method."""

    def test_extracts_valid_regiment(self) -> None:
        """Test regiment extraction with valid format."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((50, 300, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="[7-HP#123] 7th Hispanic Platoon (5 Players)",
        ):
            result = service._extract_regiment(img)
            assert result == "[7-HP#123] 7th Hispanic Platoon"

    def test_returns_none_for_invalid_format(self) -> None:
        """Test regiment returns None for invalid format."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((50, 300, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="No Regiment",
        ):
            result = service._extract_regiment(img)
            assert result is None


class TestDetectFaction:
    """Tests for _detect_faction method."""

    def test_returns_none_when_no_icons_loaded(self) -> None:
        """Test faction detection returns None when icons not loaded."""
        settings = get_settings()
        service = OCRService(settings)
        service.colonial_icon = None
        service.warden_icon = None
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        result = service._detect_faction(img)
        assert result is None

    def test_detects_colonial_faction(self) -> None:
        """Test faction detection when colonial icon matches."""
        settings = get_settings()
        service = OCRService(settings)
        # Create a simple icon pattern
        icon = np.zeros((50, 50, 3), dtype=np.uint8)
        icon[10:40, 10:40] = [0, 128, 0]  # Green square
        service.colonial_icon = icon
        service.warden_icon = None

        # Create an image with the same pattern
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = [0, 128, 0]  # Same green square

        result = service._detect_faction(img)
        assert result == Faction.COLONIAL

    def test_detects_warden_faction(self) -> None:
        """Test faction detection when warden icon matches."""
        settings = get_settings()
        service = OCRService(settings)
        # Create a simple icon pattern
        icon = np.zeros((50, 50, 3), dtype=np.uint8)
        icon[10:40, 10:40] = [128, 0, 0]  # Blue square
        service.colonial_icon = None
        service.warden_icon = icon

        # Create an image with the same pattern
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = [128, 0, 0]  # Same blue square

        result = service._detect_faction(img)
        assert result == Faction.WARDEN

    def test_returns_none_when_no_match(self) -> None:
        """Test faction detection returns None when no icon matches."""
        settings = get_settings()
        service = OCRService(settings)
        # Create icons with distinct patterns
        colonial_icon = np.zeros((50, 50, 3), dtype=np.uint8)
        colonial_icon[10:40, 10:40] = [0, 128, 0]  # Green
        warden_icon = np.zeros((50, 50, 3), dtype=np.uint8)
        warden_icon[10:40, 10:40] = [128, 0, 0]  # Blue
        service.colonial_icon = colonial_icon
        service.warden_icon = warden_icon

        # Create a completely different image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White

        result = service._detect_faction(img)
        assert result is None

    def test_handles_template_larger_than_image(self) -> None:
        """Test faction detection handles templates larger than image."""
        settings = get_settings()
        service = OCRService(settings)
        # Create a large icon
        icon = np.zeros((100, 100, 3), dtype=np.uint8)
        service.colonial_icon = icon
        service.warden_icon = None

        # Create a small image
        img = np.zeros((20, 30, 3), dtype=np.uint8)

        result = service._detect_faction(img)
        # Should handle gracefully without error
        assert result is None or result == Faction.COLONIAL


class TestExtractShardData:
    """Tests for _extract_shard_data method."""

    def test_extracts_shard_and_time(self) -> None:
        """Test shard data extraction."""
        settings = get_settings()
        service = OCRService(settings)
        # Image size must be large enough for shard region detection
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="Colonial Home Region\nDay 288, 2139 Hours\nABLE\nMap intelligence",
        ):
            result = service._extract_shard_data(image=img, profile_height=35)
            assert result["shard"] == "ABLE"
            assert result["ingame_time"] == "288, 21:39"

    def test_handles_missing_lines(self) -> None:
        """Test shard extraction with missing lines."""
        settings = get_settings()
        service = OCRService(settings)
        # Image size must be large enough for shard region detection
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            return_value="Some text only",
        ):
            result = service._extract_shard_data(image=img, profile_height=35)
            assert result["shard"] is None
            assert result["ingame_time"] is None

    def test_returns_none_for_small_image(self) -> None:
        """Test shard extraction returns None values for images too small."""
        settings = get_settings()
        service = OCRService(settings)
        # Small image that won't have a shard region
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255

        result = service._extract_shard_data(image=img, profile_height=35)
        assert result["shard"] is None
        assert result["ingame_time"] is None


class TestExtractProfileData:
    """Tests for extract_profile_data method."""

    def test_extracts_all_profile_fields(self) -> None:
        """Test profile data extraction."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((500, 800, 3), dtype=np.uint8) * 128
        boxes: list[Box] = [
            (10, 10, 150, 35),  # username
            (170, 10, 50, 35),  # icon
            (230, 10, 100, 35),  # level
            (340, 100, 200, 25),  # regiment
        ]

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "PlayerName"
            if call_count[0] == 2:
                return "25"
            return "[TAG#123] Regiment Name"

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            with patch.object(service, "_detect_faction", return_value=Faction.COLONIAL):
                result = service.extract_profile_data(img, boxes)
                assert result["name"] == "PlayerName"
                assert result["faction"] == Faction.COLONIAL

    def test_scales_small_boxes(self) -> None:
        """Test profile data extraction scales small boxes."""
        settings = get_settings()
        service = OCRService(settings)
        img = np.ones((500, 800, 3), dtype=np.uint8) * 128
        # Use smaller box height (20) which is less than PROFILE_REF_HEIGHT (35)
        # This should trigger the scaling logic
        boxes: list[Box] = [
            (10, 10, 100, 20),  # username - small height triggers scaling
            (120, 10, 35, 20),  # icon
            (165, 10, 70, 20),  # level
            (245, 70, 140, 15),  # regiment
        ]

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "ScaledPlayer"
            if call_count[0] == 2:
                return "10"
            return "[SCALED#1] Scaled Regiment"

        with patch(
            "verification_ocr.services.ocr.service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            with patch.object(service, "_detect_faction", return_value=Faction.WARDEN):
                result = service.extract_profile_data(img, boxes)
                assert result["name"] == "ScaledPlayer"
                assert result["faction"] == Faction.WARDEN
                assert result["level"] == 10


class TestVerify:
    """Tests for verify method."""

    def _create_test_image(self, width: int = 1920, height: int = 1080) -> bytes:
        """Create a test image."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def test_raises_for_invalid_image1(self) -> None:
        """Test that invalid image1 raises ValueError."""
        settings = get_settings()
        service = OCRService(settings)
        img2_bytes = self._create_test_image()
        with pytest.raises(ValueError, match="Failed to decode"):
            service.verify(b"not an image", img2_bytes)

    def test_raises_for_invalid_image2(self) -> None:
        """Test that invalid image2 raises ValueError."""
        settings = get_settings()
        service = OCRService(settings)
        img1_bytes = self._create_test_image()
        with pytest.raises(ValueError, match="Failed to decode"):
            service.verify(img1_bytes, b"not an image")

    def test_raises_when_no_profile_detected(self) -> None:
        """Test that missing profile raises ValueError."""
        settings = get_settings()
        service = OCRService(settings)
        img1_bytes = self._create_test_image()
        img2_bytes = self._create_test_image()

        with patch.object(service, "detect_profile_boxes", return_value=None):
            with pytest.raises(ValueError, match="Could not detect profile"):
                service.verify(img1_bytes, img2_bytes)

    def test_returns_verification_on_success(self) -> None:
        """Test successful verification returns Verification object."""
        settings = get_settings()
        service = OCRService(settings)
        img1_bytes = self._create_test_image()
        img2_bytes = self._create_test_image()

        mock_boxes: list[Box] = [
            (10, 10, 150, 35),
            (170, 10, 50, 35),
            (230, 10, 100, 35),
            (340, 100, 200, 25),
        ]

        profile_call_count = [0]

        def mock_detect_profile(*args: Any) -> list[Box] | None:
            profile_call_count[0] += 1
            return mock_boxes if profile_call_count[0] == 1 else None

        with patch.object(service, "detect_profile_boxes", side_effect=mock_detect_profile):
            with patch.object(
                service,
                "extract_profile_data",
                return_value=ProfileData(
                    name="TestPlayer",
                    faction=Faction.COLONIAL,
                    level=25,
                    regiment="[TEST#1] Test Regiment",
                ),
            ):
                with patch.object(
                    service,
                    "_extract_shard_data",
                    return_value=ShardData(shard="ABLE", ingame_time="288, 21:39"),
                ):
                    result = service.verify(img1_bytes, img2_bytes)
                    assert isinstance(result, Verification)
                    assert result.name == "TestPlayer"
                    assert result.faction == Faction.COLONIAL
                    assert result.level == 25
                    assert result.shard == "ABLE"


class TestGetCurrentIngameTime:
    """Tests for _get_current_ingame_time function."""

    def test_returns_none_when_not_configured(self) -> None:
        """Test returns None when war service not configured."""
        with patch("verification_ocr.services.ocr.service.get_war_service") as mock_get_war:
            mock_war_service = MagicMock()
            mock_war_service.state.start_time = None
            mock_get_war.return_value = mock_war_service

            result = _get_current_ingame_time()
            assert result is None

    def test_returns_formatted_time_when_configured(self) -> None:
        """Test returns formatted time when war service configured."""
        with patch("verification_ocr.services.ocr.service.get_war_service") as mock_get_war:
            with patch(
                "verification_ocr.services.ocr.service.calculate_war_time",
                return_value=(100, 12, 30),
            ):
                mock_war_service = MagicMock()
                mock_war_service.state.start_time = 1234567890
                mock_get_war.return_value = mock_war_service

                result = _get_current_ingame_time()
                assert result == "100, 12:30"
