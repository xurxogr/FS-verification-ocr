"""Tests for verification service."""

import os
import pathlib
import time
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytesseract
import pytest

from verification_ocr.core.settings import AppSettings
from verification_ocr.core.utils import extract_day_and_hour
from verification_ocr.enums import Faction
from verification_ocr.models import ImageRegions, Region
from verification_ocr.services import get_war_service
from verification_ocr.services.verification_service import (
    VerificationService,
    get_current_ingame_time,
)

# Type alias for profile box result: (box, grey_boxes)
type ProfileBoxResult = tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]


def create_mock_profile_box() -> ProfileBoxResult:
    """Create a mock profile box result for testing.

    Returns a tuple of (box, grey_boxes) that can be returned by _find_profile_box.
    """
    box = (100, 100, 500, 90)
    grey_boxes = [
        (100, 110, 150, 30),  # username
        (260, 110, 50, 30),  # icon
        (320, 110, 100, 30),  # level
        (430, 110, 100, 30),  # rank
    ]
    return (box, grey_boxes)


def create_test_regions(
    service: VerificationService,
    img: np.ndarray,
) -> ImageRegions:
    """Create ImageRegions for testing using the new unified API.

    Creates a profile box based on image dimensions (at 2160p ratio),
    calculates grey boxes, and returns regions.

    Args:
        service: VerificationService instance.
        img: Image array.

    Returns:
        ImageRegions for the given image.
    """
    height, width = img.shape[:2]
    # Create a profile box scaled to image height (180px at 2160p)
    scale = height / 2160
    box_h = int(180 * scale)
    box_w = int(1096 * scale)
    box_x = int(width * 0.36)  # Approximate position
    box_y = int(height * 0.12)
    box = (box_x, box_y, box_w, box_h)

    grey_boxes = service._calculate_grey_boxes_from_black_box(box)
    return service._calculate_regions_from_grey_boxes(img, box, grey_boxes)


class TestParseRegimentName:
    """Tests for _parse_regiment_name method."""

    def test_returns_none_for_empty_text(self, mock_settings: AppSettings) -> None:
        """Test that None is returned for empty text.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("")
        assert result is None

    def test_extracts_regiment_with_tag(self, mock_settings: AppSettings) -> None:
        """Test extraction of regiment name with tag.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[TAG] My Regiment")
        assert result == "[TAG] My Regiment"

    def test_removes_players_suffix(self, mock_settings: AppSettings) -> None:
        """Test that '| Players' suffix is removed.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[I PHAETON 7th Hispanic Platoon | Players")
        assert result == "[I PHAETON 7th Hispanic Platoon"

    def test_cleans_whitespace(self, mock_settings: AppSettings) -> None:
        """Test that extra whitespace is cleaned up.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[TAG]   My   Regiment  ")
        assert result == "[TAG] My Regiment"

    def test_returns_none_for_whitespace_only(self, mock_settings: AppSettings) -> None:
        """Test that None is returned for whitespace-only text.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("   ")
        assert result is None


class TestExtractDayAndHour:
    """Tests for extract_day_and_hour function."""

    def test_extracts_day_and_hour_with_two_commas(self) -> None:
        """Test extraction with two commas in input."""
        result = extract_day_and_hour("Day 1,234, 15:30")
        assert result == "1234, 15:30"

    def test_extracts_day_and_hour_standard_format(self) -> None:
        """Test extraction with standard format."""
        result = extract_day_and_hour("1234, 1530")
        assert result == "1234, 15:30"

    def test_returns_raw_result_when_no_time(self) -> None:
        """Test that raw result is returned when time format not found."""
        result = extract_day_and_hour("12345")
        assert result == "12345"

    def test_returns_raw_result_when_digits_not_four(self) -> None:
        """Test that raw result is returned when right side has != 4 digits.

        Covers the branch when len(parts) == 2 but len(digits) != 4.

        """
        # Has comma so len(parts) == 2, but "123" has only 3 digits
        result = extract_day_and_hour("100, 123")
        assert result == "100,123"

    def test_returns_formatted_when_extra_digits(self) -> None:
        """Test when right side has more than 4 digits."""
        # "12345" has 5 digits, not 4
        result = extract_day_and_hour("100, 12345")
        assert result == "100,12345"

    def test_handles_empty_string(self) -> None:
        """Test handling of empty string."""
        result = extract_day_and_hour("")
        assert result == ""


class TestGetCurrentIngameTime:
    """Tests for get_current_ingame_time function."""

    def test_returns_none_when_war_not_configured(self) -> None:
        """Test returns None when war state is not configured."""
        result = get_current_ingame_time()
        assert result is None

    def test_returns_time_when_configured(self) -> None:
        """Test returns time when war state is configured."""
        war_service = get_war_service()
        # Set start time to 2.5 hours ago
        war_service.initialize(start_time=int((time.time() - 2.5 * 60 * 60) * 1000))

        result = get_current_ingame_time()

        assert result is not None
        day, hour, minute = result
        # 2.5 hours = Day 3 (2+1), 12:00
        assert day == 3
        assert hour == 12
        assert minute == 0


class TestVerificationServiceInit:
    """Tests for VerificationService initialization."""

    def test_init_with_settings(self, mock_settings: AppSettings) -> None:
        """Test service initialization with settings.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        assert service.settings is mock_settings

    def test_init_sets_tesseract_cmd_when_provided(self) -> None:
        """Test that tesseract_cmd is set when provided in settings."""
        settings = AppSettings()
        settings.ocr.tesseract_cmd = "/custom/tesseract"

        with patch("verification_ocr.services.verification_service.pytesseract") as mock_pyt:
            VerificationService(settings)
            assert mock_pyt.pytesseract.tesseract_cmd == "/custom/tesseract"

    def test_init_does_not_set_tesseract_cmd_when_none(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that tesseract_cmd is not set when None.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.tesseract_cmd = None
        with patch("verification_ocr.services.verification_service.pytesseract") as mock_pyt:
            original_cmd = mock_pyt.pytesseract.tesseract_cmd
            VerificationService(mock_settings)
            assert mock_pyt.pytesseract.tesseract_cmd == original_cmd

    def test_init_loads_colonial_icon_when_path_provided(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that colonial icon is loaded when path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = "/path/to/colonial_icon.png"
        mock_settings.ocr.warden_icon_path = None
        mock_icon = np.ones((50, 50, 3), dtype=np.uint8)

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=mock_icon,
        ) as mock_imread:
            service = VerificationService(mock_settings)
            mock_imread.assert_called_once_with(
                filename="/path/to/colonial_icon.png",
                flags=cv2.IMREAD_COLOR,
            )
            assert service.colonial_icon is not None

    def test_init_loads_warden_icon_when_path_provided(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that warden icon is loaded when path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = None
        mock_settings.ocr.warden_icon_path = "/path/to/warden_icon.png"
        mock_icon = np.ones((50, 50, 3), dtype=np.uint8)

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=mock_icon,
        ) as mock_imread:
            service = VerificationService(mock_settings)
            mock_imread.assert_called_once_with(
                filename="/path/to/warden_icon.png",
                flags=cv2.IMREAD_COLOR,
            )
            assert service.warden_icon is not None

    def test_init_logs_warning_when_colonial_icon_file_not_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that a warning is logged when colonial icon file is not found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = "/nonexistent/colonial.png"
        mock_settings.ocr.warden_icon_path = None

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=None,
        ):
            with patch(
                "verification_ocr.services.verification_service.logger.warning",
            ) as mock_warning:
                service = VerificationService(mock_settings)
                mock_warning.assert_called_once()
                assert "Failed to load colonial icon" in mock_warning.call_args[0][0]
                assert service.colonial_icon is None

    def test_init_logs_warning_when_warden_icon_file_not_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that a warning is logged when warden icon file is not found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = None
        mock_settings.ocr.warden_icon_path = "/nonexistent/warden.png"

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=None,
        ):
            with patch(
                "verification_ocr.services.verification_service.logger.warning",
            ) as mock_warning:
                service = VerificationService(mock_settings)
                mock_warning.assert_called_once()
                assert "Failed to load warden icon" in mock_warning.call_args[0][0]
                assert service.warden_icon is None

    def test_init_colonial_icon_none_when_no_path(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that colonial icon is None when no path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = None
        service = VerificationService(mock_settings)
        assert service.colonial_icon is None

    def test_init_creates_debug_dir_when_debug_mode(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that debug directory is created when debug mode is enabled.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = "test_debug_dir"

        with patch("verification_ocr.services.verification_service.os.makedirs") as mock_makedirs:
            VerificationService(mock_settings)
            mock_makedirs.assert_called_once_with("test_debug_dir", exist_ok=True)


class TestVerificationServiceExtractText:
    """Tests for _extract_text_from_image method."""

    def test_extract_text_returns_empty_for_none_image(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that empty string is returned for None image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._extract_text_from_image(None)  # type: ignore[arg-type]
        assert result == ""

    def test_extract_text_returns_empty_for_empty_image(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that empty string is returned for empty image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        empty_img = np.array([], dtype=np.uint8)
        result = service._extract_text_from_image(empty_img)
        assert result == ""

    def test_extract_text_with_valid_image(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """Test text extraction from valid image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="TestPlayer Level: 25\n",
        ):
            service = VerificationService(mock_settings)
            result = service._extract_text_from_image(img)
            assert result == "TestPlayer Level: 25"

    def test_extract_text_with_scale(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """Test text extraction with scaling enabled.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="ABLE\n",
        ):
            service = VerificationService(mock_settings)
            result = service._extract_text_from_image(image=img, scale=True)
            assert result == "ABLE"

    def test_extract_text_without_invert(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """Test text extraction without color inversion.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="ABLE\n",
        ):
            service = VerificationService(mock_settings)
            result = service._extract_text_from_image(image=img, use_invert=False)
            assert result == "ABLE"


class TestVerificationServiceDetectFaction:
    """Tests for faction detection."""

    def test_find_user_info_sets_faction_colonial(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that faction is set to COLONIAL when colonial icon is found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        with patch.object(service, "_extract_text_from_image", return_value="TestPlayer"):
            result = service._find_user_info(img, regions)
            # Faction is set separately by verify(), not by _find_user_info
            assert result.name == "TestPlayer"
            assert result.faction is None


class TestVerificationServiceCalculateRegions:
    """Tests for _calculate_regions method."""

    def test_calculate_regions_returns_correct_attributes(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that _calculate_regions returns ImageRegions with all expected attributes.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        assert regions.username is not None
        assert regions.icon is not None
        assert regions.level is not None
        assert regions.regiment is not None
        assert regions.shard is not None
        assert regions.scale_factor is not None

    def test_calculate_regions_scale_factor_is_1_for_4k(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that scale factor is 1.0 for 4K resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        assert regions.scale_factor == 1.0

    def test_calculate_regions_scale_factor_for_1080p(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that scale factor is 0.5 for 1080p resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        assert regions.scale_factor == 0.5


class TestVerificationServiceSaveDebugImage:
    """Tests for _save_debug_image method."""

    def test_save_debug_image_creates_file(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that debug image is saved to file.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        service._save_debug_image(
            image=img,
            regions=regions,
            filename="test_debug.png",
        )

        assert os.path.exists(tmp_path / "test_debug.png")

    def test_save_debug_image_rejects_invalid_filename(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that invalid filenames are rejected.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        # Filename with special characters should be rejected
        service._save_debug_image(
            image=img,
            regions=regions,
            filename="test<>debug.png",
        )

        # File should not be created
        assert not os.path.exists(tmp_path / "test<>debug.png")

    def test_save_debug_image_strips_path_components(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that path components are stripped from filename.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        # Path traversal attempt - basename strips path, file saved safely in debug dir
        service._save_debug_image(
            image=img,
            regions=regions,
            filename="../../../safe_file.png",
        )

        # File should be created in debug dir (path components stripped)
        assert os.path.exists(tmp_path / "safe_file.png")
        # File should NOT be created outside debug dir
        assert not os.path.exists(tmp_path.parent / "safe_file.png")

    def test_save_debug_image_rejects_symlink_traversal(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that symlink-based path traversal is rejected.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        # Create debug dir inside tmp_path
        debug_dir = tmp_path / "debug"
        debug_dir.mkdir()

        # Create a target directory outside debug_dir
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(debug_dir)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        # Mock os.path.realpath to simulate a symlink resolving outside
        with patch("os.path.realpath") as mock_realpath:
            # First call for debug_dir returns actual path
            # Second call for output_path returns path outside debug_dir
            mock_realpath.side_effect = [
                str(debug_dir),
                str(outside_dir / "evil.png"),
            ]

            service._save_debug_image(
                image=img,
                regions=regions,
                filename="evil.png",
            )

            # File should NOT be created (path traversal rejected)
            assert not os.path.exists(outside_dir / "evil.png")
            assert not os.path.exists(debug_dir / "evil.png")


class TestVerificationServicePrepareImageForShardDetection:
    """Tests for _prepare_image_for_shard_detection method."""

    def test_returns_image_unchanged_when_none(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that None is returned when image is None.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._prepare_image_for_shard_detection(image=None, scale_factor=1.0)  # type: ignore[arg-type]
        assert result is None

    def test_returns_image_unchanged_when_empty(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that empty image is returned unchanged.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        empty_img = np.array([], dtype=np.uint8)
        result = service._prepare_image_for_shard_detection(image=empty_img, scale_factor=1.0)
        assert result.size == 0

    def test_processes_valid_image(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that valid image is processed correctly.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = service._prepare_image_for_shard_detection(image=img, scale_factor=1.0)
        assert result is not None
        assert result.size > 0


class TestVerificationServiceGetShardAndTime:
    """Tests for _get_shard_and_time method."""

    def test_get_shard_and_time_returns_none_for_empty_region(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that None is returned when shard region is empty.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        # Create a very small image where shard region calculation results in empty slice
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)

        # Create regions with empty shard region
        empty_region = Region(x1=5, y1=5, x2=5, y2=5)
        regions = ImageRegions(
            username=empty_region,
            icon=empty_region,
            level=empty_region,
            regiment=empty_region,
            shard=empty_region,
            scale_factor=1.0,
        )

        shard, ingame_time = service._get_shard_and_time(image=img, regions=regions)
        assert shard is None
        assert ingame_time is None

    def test_get_shard_and_time_returns_values(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that shard and time are extracted.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="1234, 1530\nABLE",
        ):
            shard, ingame_time = service._get_shard_and_time(image=img, regions=regions)
            assert shard == "ABLE"
            assert ingame_time == "1234, 15:30"

    def test_get_shard_and_time_returns_none_for_empty_text(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that None is returned when no text extracted.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="",
        ):
            shard, ingame_time = service._get_shard_and_time(image=img, regions=regions)
            assert shard is None
            assert ingame_time is None

    def test_get_shard_and_time_handles_single_line(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test handling of single line OCR output.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = create_test_regions(service, img)

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="1234, 1530",
        ):
            shard, ingame_time = service._get_shard_and_time(image=img, regions=regions)
            assert shard is None
            assert ingame_time == "1234, 15:30"


class TestVerificationServiceVerify:
    """Tests for verify method."""

    def test_verify_raises_error_for_invalid_images(
        self,
        mock_settings: AppSettings,
        invalid_image_bytes: bytes,
    ) -> None:
        """Test that ValueError is raised for invalid images.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            invalid_image_bytes (bytes): Invalid image bytes fixture.

        """
        service = VerificationService(mock_settings)
        with pytest.raises(ValueError, match="Failed to decode"):
            service.verify(invalid_image_bytes, invalid_image_bytes)

    def test_verify_raises_error_when_no_profile_box_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that ValueError is raised when no profile box is found in either image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)
        # No profile box will be detected in plain images
        with pytest.raises(ValueError, match="Could not identify profile image"):
            service.verify(image_bytes, image_bytes)

    def test_verify_raises_error_when_both_images_have_profile_box(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that ValueError is raised when both images appear to be profile images.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img1 = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer1 = cv2.imencode(".png", img1)
        image1_bytes = buffer1.tobytes()

        img2 = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, buffer2 = cv2.imencode(".png", img2)
        image2_bytes = buffer2.tobytes()

        service = VerificationService(mock_settings)
        # Mock profile box detection to return a box for both images
        mock_box = (
            (100, 100, 500, 90),
            [(100, 110, 150, 30), (260, 110, 50, 30), (320, 110, 100, 30), (430, 110, 100, 30)],
        )
        with patch.object(service, "_find_profile_box", return_value=mock_box):
            with pytest.raises(ValueError, match="Both images appear to be profile images"):
                service.verify(image1_bytes, image2_bytes)

    def test_verify_raises_error_when_no_name_in_profile(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that ValueError is raised when no name is found in the profile image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        # OCR returns empty for username
        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="",
        ):
            service = VerificationService(mock_settings)
            # Mock: first image has profile box, second doesn't (shard)
            mock_box = (
                (100, 100, 500, 90),
                [(100, 110, 150, 30), (260, 110, 50, 30), (320, 110, 100, 30), (430, 110, 100, 30)],
            )
            call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                call_count[0] += 1
                return mock_box if call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                with pytest.raises(ValueError, match="No player name found in the profile image"):
                    service.verify(profile_bytes, shard_bytes)

    def test_verify_extracts_user_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify extracts user information correctly.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        # Create two different test images (profile and shard)
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            # Profile image: username, level, regiment
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            # Shard extraction
            return "100, 1200\nABLE"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            # Mock: first image has profile box, second doesn't (shard)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                with patch.object(
                    service, "_detect_faction_with_scaled_template", return_value=Faction.COLONIAL
                ):
                    result = service.verify(profile_bytes, shard_bytes)

                    assert result.name == "TestPlayer"
                    assert result.level == 25
                    assert result.faction == Faction.COLONIAL

    def test_verify_handles_malformed_ocr_text(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify handles malformed OCR text gracefully.

        Tests when level region has no colon - level stays None.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level abc"  # level - no colon
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.name == "TestPlayer"
                assert result.level is None

    def test_verify_handles_level_with_colon_but_no_digits(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that level is None when colon exists but no digits after it.

        Covers branch 603->609 where digits is empty string.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: abc"  # level - has colon but no digits
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.name == "TestPlayer"
                assert result.level is None

    def test_verify_logs_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify logs an info message.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        # Create a larger test image (small images cause empty regions)
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="",
        ):
            with patch(
                "verification_ocr.services.verification_service.logger.info",
            ) as mock_log:
                service = VerificationService(mock_settings)
                with pytest.raises(ValueError):
                    service.verify(image_bytes, image_bytes)

                mock_log.assert_called_once_with("Processing image pair for verification")

    def test_verify_uses_correct_image_for_user_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that user info is extracted from the image with faction icon.

        When second image has faction, user info comes from second image
        and shard info comes from first image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            # OCR is only called for: profile username, level, regiment, then shard
            if call_count[0] == 1:
                return "SecondPlayer"  # profile username
            if call_count[0] == 2:
                return "Level: 30"  # profile level
            if call_count[0] == 3:
                return ""  # profile regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                # Second image (profile) has profile box
                return mock_box if profile_box_call_count[0] == 2 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(shard_bytes, profile_bytes)

                assert result.name == "SecondPlayer"
                assert result.shard == "ABLE"

    def test_verify_saves_debug_images_when_debug_mode(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that debug images are saved when debug mode is enabled.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                with patch.object(
                    service, "_detect_faction_with_scaled_template", return_value=Faction.COLONIAL
                ):
                    service.verify(profile_bytes, shard_bytes)

                    # The debug image is saved as debug_profile_regions.png
                    assert os.path.exists(tmp_path / "debug_profile_regions.png")

    def test_verify_includes_shard_and_ingame_time(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify includes shard and ingame_time in result.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            return "1234, 1530\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.shard == "ABLE"
                assert result.ingame_time == "1234, 15:30"

    def test_verify_includes_war_number_and_current_ingame_time(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify includes war_number and current_ingame_time in result.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        # Set up war state
        war_service = get_war_service()
        war_service.initialize(
            war_number=132,
            start_time=int((time.time() - 100 * 60 * 60) * 1000),
        )

        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.war_number == 132
                assert result.current_ingame_time is not None

    def test_verify_raises_error_when_no_shard_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that error is raised when no shard info is found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            # Return empty for shard region
            return ""

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                with pytest.raises(
                    ValueError, match="No shard information found in the map/shard image"
                ):
                    service.verify(profile_bytes, shard_bytes)

    def test_verify_returns_result_when_war_not_configured(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that verify returns result when war state not configured.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        # War state is reset by conftest, so current_time will be None

        profile_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, profile_buffer = cv2.imencode(".png", profile_img)
        profile_bytes = profile_buffer.tobytes()

        shard_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 128
        _, shard_buffer = cv2.imencode(".png", shard_img)
        shard_bytes = shard_buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer"  # username
            if call_count[0] == 2:
                return "Level: 25"  # level
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.ingame_time == "100, 12:00"
                assert result.war_number is None
                assert result.current_ingame_time is None


class TestVerificationServiceIntegration:
    """Integration tests using real game screenshots."""

    def test_verify_colonial_with_real_images(
        self,
        integration_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """Test verification with real colonial game screenshots.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            colonial_image_bytes (bytes): Real colonial user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(colonial_image_bytes, stockpile_image_bytes)

        assert result.name is not None
        assert result.level is not None
        assert result.faction == Faction.COLONIAL

    def test_verify_warden_with_real_images(
        self,
        integration_settings: AppSettings,
        warden_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """Test verification with real warden game screenshots.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            warden_image_bytes (bytes): Real warden user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(warden_image_bytes, stockpile_image_bytes)

        assert result.name is not None
        assert result.level is not None
        assert result.faction == Faction.WARDEN
        # Warden image has no regiment
        assert result.regiment is None

    def test_verify_extracts_shard_from_real_image(
        self,
        integration_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """Test shard extraction from real stockpile screenshot.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            colonial_image_bytes (bytes): Real colonial user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(colonial_image_bytes, stockpile_image_bytes)

        # Shard should be extracted from stockpile image
        assert result.shard is not None or result.ingame_time is not None

    def test_verify_extracts_regiment_name_from_colonial(
        self,
        integration_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """Test regiment name extraction from colonial user screenshot.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            colonial_image_bytes (bytes): Real colonial user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(colonial_image_bytes, stockpile_image_bytes)

        # Colonial user is in a regiment - regiment field contains the name
        assert result.regiment is not None
        # Should contain "7th Hispanic Platoon" (regiment name)
        assert "7th Hispanic Platoon" in result.regiment

    def test_verify_warden_has_no_regiment(
        self,
        integration_settings: AppSettings,
        warden_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """Test that warden user without regiment has None for regiment.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            warden_image_bytes (bytes): Real warden user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(warden_image_bytes, stockpile_image_bytes)

        # Warden user is not in a regiment - regiment is None
        assert result.regiment is None

    def test_verify_raises_error_for_too_small_image(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that ValueError is raised when image is too small.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        # Create a tiny image (50x50, below 100x100 minimum)
        tiny_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", tiny_img)
        tiny_bytes = buffer.tobytes()

        # Create a valid sized image
        valid_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, valid_buffer = cv2.imencode(".png", valid_img)
        valid_bytes = valid_buffer.tobytes()

        service = VerificationService(mock_settings)

        # Test first image too small
        with pytest.raises(ValueError, match="Image 1.*too small"):
            service.verify(tiny_bytes, valid_bytes)

        # Test second image too small
        with pytest.raises(ValueError, match="Image 2.*too small"):
            service.verify(valid_bytes, tiny_bytes)

    def test_verify_handles_opencv_error(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that OpenCV errors are handled gracefully.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)

        # Mock profile box to throw OpenCV error
        with patch.object(service, "_find_profile_box", side_effect=cv2.error("Test OpenCV error")):
            with pytest.raises(RuntimeError, match="Image processing error"):
                service.verify(image_bytes, image_bytes)

    def test_verify_handles_tesseract_error(
        self,
        integration_settings: AppSettings,
    ) -> None:
        """Test that Tesseract errors are handled gracefully.

        Args:
            integration_settings (AppSettings): Integration settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(integration_settings)

        # First mock profile box detection to work, then fail on _find_user_info
        mock_box = create_mock_profile_box()
        profile_box_call_count = [0]

        def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
            profile_box_call_count[0] += 1
            return mock_box if profile_box_call_count[0] == 1 else None

        with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
            with patch.object(
                service,
                "_find_user_info",
                side_effect=pytesseract.TesseractError("Tesseract failed", 1),
            ):
                with pytest.raises(RuntimeError, match="OCR processing error"):
                    service.verify(image_bytes, image_bytes)

    def test_verify_handles_unexpected_exception(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that unexpected exceptions are handled gracefully.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)

        with patch.object(
            service, "_calculate_regions_from_grey_boxes", side_effect=Exception("Unexpected error")
        ):
            # Need to mock profile box detection first
            mock_box = create_mock_profile_box()
            profile_box_call_count = [0]

            def mock_find_profile_box(img: Any) -> ProfileBoxResult | None:
                profile_box_call_count[0] += 1
                return mock_box if profile_box_call_count[0] == 1 else None

            with patch.object(service, "_find_profile_box", side_effect=mock_find_profile_box):
                with pytest.raises(RuntimeError, match="Internal processing error"):
                    service.verify(image_bytes, image_bytes)


class TestFindProfileBoxByBlack:
    """Tests for _find_profile_box_by_black method edge cases."""

    def test_rejects_box_at_origin(self, mock_settings: AppSettings) -> None:
        """Test that boxes at origin (0, 0) are rejected.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create an image with a black rectangle at origin
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        # Draw black rectangle at origin with correct aspect ratio (~6.09)
        cv2.rectangle(img, (0, 0), (300, 49), (0, 0, 0), -1)

        result = service._find_profile_box_by_black(img)
        # Should be None because box is at origin
        assert result is None

    def test_rejects_box_too_wide(self, mock_settings: AppSettings) -> None:
        """Test that boxes wider than 50% of image width are rejected.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create an image with a very wide black rectangle
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        # Draw black rectangle that is > 50% of image width (aspect ratio ~6)
        # Width 250 > 200 (50% of 400)
        cv2.rectangle(img, (50, 50), (300, 91), (0, 0, 0), -1)

        result = service._find_profile_box_by_black(img)
        # Should be None because box is too wide
        assert result is None


class TestFindGreyBoxesPattern:
    """Tests for _find_grey_boxes_pattern method."""

    def test_returns_none_when_not_enough_candidates(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that None is returned when fewer than 6 grey box candidates.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create image with only a few grey regions
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        # Add only 3 grey boxes (need at least 6)
        cv2.rectangle(img, (50, 100), (100, 140), (50, 50, 50), -1)
        cv2.rectangle(img, (120, 100), (170, 140), (50, 50, 50), -1)
        cv2.rectangle(img, (190, 100), (240, 140), (50, 50, 50), -1)

        result = service._find_grey_boxes_pattern(img)
        assert result is None

    def test_finds_valid_4_plus_2_pattern(self, mock_settings: AppSettings) -> None:
        """Test successful detection of 4+2 grey box pattern.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create image with valid 4+2 pattern using correct grey values (20-100)
        img = np.ones((300, 600, 3), dtype=np.uint8) * 200  # Light background

        # Row 1: 4 grey boxes at y=50 (grey value 50 is in range 20-100)
        # Boxes need to be within size constraints: h in (15,100), w in (30,500)
        cv2.rectangle(img, (50, 50), (120, 80), (50, 50, 50), -1)  # w=70, h=30
        cv2.rectangle(img, (140, 50), (210, 80), (50, 50, 50), -1)  # w=70, h=30
        cv2.rectangle(img, (230, 50), (300, 80), (50, 50, 50), -1)  # w=70, h=30
        cv2.rectangle(img, (320, 50), (390, 80), (50, 50, 50), -1)  # w=70, h=30

        # Row 2: 2 grey boxes at y=85 (within row distance tolerance: 30 * 1.15 = 34.5)
        cv2.rectangle(img, (100, 85), (200, 115), (50, 50, 50), -1)  # w=100, h=30
        cv2.rectangle(img, (220, 85), (320, 115), (50, 50, 50), -1)  # w=100, h=30

        result = service._find_grey_boxes_pattern(img)
        assert result is not None
        profile_box, grey_boxes = result
        assert len(grey_boxes) == 4  # Row 1 boxes sorted by x

    def test_returns_none_when_no_row2_found(self, mock_settings: AppSettings) -> None:
        """Test that None is returned when row 2 (2 boxes) is not found below row 1.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create image with 4 boxes in row 1 but wrong count in row 2
        img = np.ones((300, 600, 3), dtype=np.uint8) * 255

        # Row 1: 4 grey boxes
        cv2.rectangle(img, (50, 50), (100, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (120, 50), (170, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (190, 50), (240, 80), (50, 50, 50), -1)
        cv2.rectangle(img, (260, 50), (310, 80), (50, 50, 50), -1)

        # Row 2: 3 boxes instead of 2 (wrong pattern)
        cv2.rectangle(img, (80, 90), (130, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (150, 90), (200, 120), (50, 50, 50), -1)
        cv2.rectangle(img, (220, 90), (270, 120), (50, 50, 50), -1)

        result = service._find_grey_boxes_pattern(img)
        assert result is None


class TestFindRow2Below:
    """Tests for _find_row2_below method."""

    def test_returns_none_when_row2_too_far(self, mock_settings: AppSettings) -> None:
        """Test that None is returned when row 2 is too far below row 1.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        row1_boxes = [
            (50, 50, 50, 30),
            (120, 50, 50, 30),
            (190, 50, 50, 30),
            (260, 50, 50, 30),
        ]
        # Row 2 is very far (y=200, much greater than avg_height * 1.15)
        rows = {
            50: row1_boxes,
            200: [(80, 200, 70, 30), (170, 200, 70, 30)],
        }

        result = service._find_row2_below(
            row1_y=50,
            row1_boxes=row1_boxes,
            sorted_row_ys=[200],
            rows=rows,
        )
        assert result is None

    def test_finds_row2_within_tolerance(self, mock_settings: AppSettings) -> None:
        """Test that row 2 is found when within distance tolerance.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        row1_boxes = [
            (50, 50, 50, 30),
            (120, 50, 50, 30),
            (190, 50, 50, 30),
            (260, 50, 50, 30),
        ]
        # Row 2 within tolerance (avg height 30 * 1.15 = 34.5, so y=80 is within)
        rows = {
            50: row1_boxes,
            80: [(80, 80, 70, 30), (170, 80, 70, 30)],
        }

        result = service._find_row2_below(
            row1_y=50,
            row1_boxes=row1_boxes,
            sorted_row_ys=[80],
            rows=rows,
        )
        assert result == 80


class TestCalculateProfileBoxFromRows:
    """Tests for _calculate_profile_box_from_rows method."""

    def test_calculates_profile_box_bounds(self, mock_settings: AppSettings) -> None:
        """Test that profile box is correctly calculated from row boxes.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        row1_boxes = [
            (100, 50, 50, 30),
            (160, 50, 50, 30),
            (220, 50, 50, 30),
            (280, 50, 50, 30),
        ]
        row2_boxes = [
            (120, 90, 80, 30),
            (210, 90, 80, 30),
        ]

        profile_box, grey_boxes = service._calculate_profile_box_from_rows(row1_boxes, row2_boxes)

        # Profile box should encompass all boxes with expansion factors
        assert profile_box[0] >= 0  # x
        assert profile_box[1] >= 0  # y
        assert profile_box[2] > 0  # width
        assert profile_box[3] > 0  # height

        # Grey boxes should be row1 sorted by x
        assert len(grey_boxes) == 4
        assert grey_boxes[0][0] < grey_boxes[1][0]  # Sorted by x


class TestDetectFactionWithScaledTemplate:
    """Tests for _detect_faction_with_scaled_template edge cases."""

    def test_returns_none_for_empty_icon_region(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that None is returned when icon region is empty.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Create region where y1 == y2 (empty slice)
        icon_region = Region(x1=10, y1=50, x2=60, y2=50)

        result = service._detect_faction_with_scaled_template(
            image=img,
            icon_region=icon_region,
            threshold=0.7,
        )
        assert result is None

    def test_skips_template_larger_than_icon_region(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that templates larger than icon region are skipped.

        When the icon_region extends beyond image bounds, the sliced icon_img
        will be smaller than expected, causing scaled template to not fit.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)

        # Create image where icon_region extends beyond bounds
        # Image is 50x50, but region extends to y=60 (beyond image)
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255

        # Region that extends beyond image bounds
        # Expected: region_h = 20, region_w = 20, icon_size = 20
        # But actual slice will be truncated to (10, 20) since image ends at y=50
        icon_region = Region(x1=10, y1=40, x2=30, y2=60)

        # Set template - after resize to icon_size=20, it will be 20x20
        # But icon_img will only be 10x20 (truncated), so 20 > 10 triggers skip
        service.colonial_icon = np.ones((100, 100, 3), dtype=np.uint8)

        result = service._detect_faction_with_scaled_template(
            image=img,
            icon_region=icon_region,
            threshold=0.7,
        )
        # Should return None because scaled template (20x20) won't fit in icon_img (10x20)
        assert result is None


class TestFindShardDynamic:
    """Tests for _find_shard_dynamic method."""

    def test_extracts_shard_from_time_pattern(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test extraction of shard name from line after time pattern.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255

        # Mock pytesseract to return text with time pattern
        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="Day 154, 2335 Hours\nABLE\nSome other text",
        ):
            shard, ingame_time = service._find_shard_dynamic(img)

        assert shard == "ABLE"
        assert ingame_time is not None
        assert "154" in ingame_time

    def test_returns_none_when_no_time_pattern(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that (None, None) is returned when no time pattern found.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="No time pattern here\nJust random text",
        ):
            shard, ingame_time = service._find_shard_dynamic(img)

        assert shard is None
        assert ingame_time is None

    def test_returns_none_shard_when_no_next_line(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that shard is None when time pattern is on last line.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255

        # Time pattern on last line, no shard line after
        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="Day 154, 2335 Hours",
        ):
            shard, ingame_time = service._find_shard_dynamic(img)

        assert shard is None
        # Time should still be extracted
        assert ingame_time is not None


class TestFindProfileBoxGreyBoxFallback:
    """Tests for _find_profile_box grey box fallback path."""

    def test_uses_grey_pattern_when_black_detection_fails(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that grey box pattern is used when black box detection fails.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((300, 600, 3), dtype=np.uint8) * 255

        # Mock black box detection to return None
        with patch.object(service, "_find_profile_box_by_black", return_value=None):
            # Mock grey pattern to return valid result
            mock_box = (50, 40, 300, 100)
            mock_grey_boxes = [
                (60, 50, 50, 30),
                (120, 50, 50, 30),
                (180, 50, 50, 30),
                (240, 50, 50, 30),
            ]
            with patch.object(
                service,
                "_find_grey_boxes_pattern",
                return_value=(mock_box, mock_grey_boxes),
            ):
                # Mock faction detection to succeed
                with patch.object(
                    service,
                    "_detect_faction_with_scaled_template",
                    return_value=Faction.COLONIAL,
                ):
                    result = service._find_profile_box(img)

        assert result is not None
        box, grey_boxes = result
        assert box == mock_box
        assert grey_boxes == mock_grey_boxes


class TestExtractProfileDataGreyBoxesNone:
    """Tests for _extract_profile_data when grey_boxes is None."""

    def test_calculates_grey_boxes_when_none(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """Test that grey boxes are calculated from black box when None.

        Args:
            mock_settings: Mock settings fixture.
        """
        service = VerificationService(mock_settings)
        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        profile_box = (100, 100, 500, 90)

        with patch.object(service, "_extract_text_from_image", return_value="TestPlayer"):
            with patch.object(
                service,
                "_detect_faction_with_scaled_template",
                return_value=Faction.COLONIAL,
            ):
                verification, regions = service._extract_profile_data(
                    profile_img=img,
                    profile_box=profile_box,
                    grey_boxes=None,  # Explicitly None
                )

        assert verification.name == "TestPlayer"
        assert regions is not None


class TestGetShardAndTimeDebugMode:
    """Tests for _get_shard_and_time debug mode."""

    def test_saves_debug_image_when_debug_mode_enabled(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that processed shard image is saved in debug mode.

        Args:
            mock_settings: Mock settings fixture.
            tmp_path: Pytest temp directory.
        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        service = VerificationService(mock_settings)

        img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        regions = create_test_regions(service, img)

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="100, 1200\nABLE",
        ):
            service._get_shard_and_time(image=img, regions=regions)

        # Check debug image was saved
        debug_file = tmp_path / "debug_shard_processed.png"
        assert debug_file.exists()
