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
from verification_ocr.enums import Faction
from verification_ocr.models import ImageRegions, Region
from verification_ocr.services import get_war_service
from verification_ocr.services.verification_service import (
    VerificationService,
    calculate_ingame_time_diff,
    extract_day_and_hour,
    get_current_ingame_time,
    parse_ingame_time,
)


class TestParseRegimentName:
    """Tests for _parse_regiment_name method."""

    def test_returns_none_for_empty_text(self, mock_settings: AppSettings) -> None:
        """
        Test that None is returned for empty text.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("")
        assert result is None

    def test_extracts_regiment_with_tag(self, mock_settings: AppSettings) -> None:
        """
        Test extraction of regiment name with tag.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[TAG] My Regiment")
        assert result == "[TAG] My Regiment"

    def test_removes_players_suffix(self, mock_settings: AppSettings) -> None:
        """
        Test that '| Players' suffix is removed.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[I PHAETON 7th Hispanic Platoon | Players")
        assert result == "[I PHAETON 7th Hispanic Platoon"

    def test_cleans_whitespace(self, mock_settings: AppSettings) -> None:
        """
        Test that extra whitespace is cleaned up.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("[TAG]   My   Regiment  ")
        assert result == "[TAG] My Regiment"

    def test_returns_none_for_whitespace_only(self, mock_settings: AppSettings) -> None:
        """
        Test that None is returned for whitespace-only text.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        result = service._parse_regiment_name("   ")
        assert result is None


class TestExtractDayAndHour:
    """Tests for extract_day_and_hour function."""

    def test_extracts_day_and_hour_with_two_commas(self) -> None:
        """
        Test extraction with two commas in input.

        """
        result = extract_day_and_hour("Day 1,234, 15:30")
        assert result == "1234, 15:30"

    def test_extracts_day_and_hour_standard_format(self) -> None:
        """
        Test extraction with standard format.

        """
        result = extract_day_and_hour("1234, 1530")
        assert result == "1234, 15:30"

    def test_returns_raw_result_when_no_time(self) -> None:
        """
        Test that raw result is returned when time format not found.

        """
        result = extract_day_and_hour("12345")
        assert result == "12345"

    def test_returns_raw_result_when_digits_not_four(self) -> None:
        """
        Test that raw result is returned when right side has != 4 digits.

        Covers the branch when len(parts) == 2 but len(digits) != 4.

        """
        # Has comma so len(parts) == 2, but "123" has only 3 digits
        result = extract_day_and_hour("100, 123")
        assert result == "100,123"

    def test_returns_formatted_when_extra_digits(self) -> None:
        """
        Test when right side has more than 4 digits.

        """
        # "12345" has 5 digits, not 4
        result = extract_day_and_hour("100, 12345")
        assert result == "100,12345"

    def test_handles_empty_string(self) -> None:
        """
        Test handling of empty string.

        """
        result = extract_day_and_hour("")
        assert result == ""


class TestParseIngameTime:
    """Tests for parse_ingame_time function."""

    def test_parses_valid_time(self) -> None:
        """
        Test parsing valid time string.

        """
        result = parse_ingame_time("267, 21:45")
        assert result == (267, 21, 45)

    def test_returns_none_for_invalid_format(self) -> None:
        """
        Test returns None for invalid format.

        """
        result = parse_ingame_time("invalid")
        assert result is None

    def test_returns_none_for_missing_time(self) -> None:
        """
        Test returns None when time part is missing.

        """
        result = parse_ingame_time("267")
        assert result is None

    def test_returns_none_for_invalid_time_format(self) -> None:
        """
        Test returns None for invalid time format.

        """
        result = parse_ingame_time("267, 2145")
        assert result is None

    def test_returns_none_for_non_numeric_values(self) -> None:
        """
        Test returns None for non-numeric values.

        """
        result = parse_ingame_time("abc, de:fg")
        assert result is None

    def test_returns_none_for_day_less_than_one(self) -> None:
        """
        Test returns None when day is less than 1.

        """
        result = parse_ingame_time("0, 12:30")
        assert result is None

    def test_returns_none_for_day_greater_than_9999(self) -> None:
        """
        Test returns None when day exceeds 9999.

        """
        result = parse_ingame_time("10000, 12:30")
        assert result is None

    def test_returns_none_for_hour_out_of_range(self) -> None:
        """
        Test returns None when hour is outside 0-23 range.

        """
        result = parse_ingame_time("267, 24:30")
        assert result is None

        result = parse_ingame_time("267, 25:30")
        assert result is None

    def test_returns_none_for_minute_out_of_range(self) -> None:
        """
        Test returns None when minute is outside 0-59 range.

        """
        result = parse_ingame_time("267, 12:60")
        assert result is None

        result = parse_ingame_time("267, 12:99")
        assert result is None


class TestCalculateIngameTimeDiff:
    """Tests for calculate_ingame_time_diff function."""

    def test_same_time_returns_zero(self) -> None:
        """
        Test that same times return zero difference.

        """
        result = calculate_ingame_time_diff(
            extracted_day=267,
            extracted_hour=21,
            current_day=267,
            current_hour=21,
        )
        assert result == 0

    def test_one_hour_difference(self) -> None:
        """
        Test one hour difference.

        """
        result = calculate_ingame_time_diff(
            extracted_day=267,
            extracted_hour=21,
            current_day=267,
            current_hour=22,
        )
        assert result == 1

    def test_one_day_difference(self) -> None:
        """
        Test one day difference.

        """
        result = calculate_ingame_time_diff(
            extracted_day=267,
            extracted_hour=0,
            current_day=268,
            current_hour=0,
        )
        assert result == 24

    def test_returns_absolute_difference(self) -> None:
        """
        Test that absolute difference is returned.

        """
        result = calculate_ingame_time_diff(
            extracted_day=268,
            extracted_hour=0,
            current_day=267,
            current_hour=0,
        )
        assert result == 24


class TestGetCurrentIngameTime:
    """Tests for get_current_ingame_time function."""

    def test_returns_none_when_war_not_configured(self) -> None:
        """
        Test returns None when war state is not configured.

        """
        result = get_current_ingame_time()
        assert result is None

    def test_returns_time_when_configured(self) -> None:
        """
        Test returns time when war state is configured.

        """
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
        """
        Test service initialization with settings.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        service = VerificationService(mock_settings)
        assert service.settings is mock_settings

    def test_init_sets_tesseract_cmd_when_provided(self) -> None:
        """
        Test that tesseract_cmd is set when provided in settings.

        """
        settings = AppSettings()
        settings.ocr.tesseract_cmd = "/custom/tesseract"

        with patch("verification_ocr.services.verification_service.pytesseract") as mock_pyt:
            VerificationService(settings)
            assert mock_pyt.pytesseract.tesseract_cmd == "/custom/tesseract"

    def test_init_does_not_set_tesseract_cmd_when_none(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that tesseract_cmd is not set when None.

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
        """
        Test that colonial icon is loaded when path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = "/path/to/colonial_icon.png"
        mock_settings.ocr.wardens_icon_path = None
        mock_icon = np.ones((50, 50, 3), dtype=np.uint8)

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=mock_icon,
        ) as mock_imread:
            service = VerificationService(mock_settings)
            mock_imread.assert_called_once_with("/path/to/colonial_icon.png", cv2.IMREAD_COLOR)
            assert service.colonial_icon is not None

    def test_init_loads_wardens_icon_when_path_provided(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that wardens icon is loaded when path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = None
        mock_settings.ocr.wardens_icon_path = "/path/to/wardens_icon.png"
        mock_icon = np.ones((50, 50, 3), dtype=np.uint8)

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=mock_icon,
        ) as mock_imread:
            service = VerificationService(mock_settings)
            mock_imread.assert_called_once_with("/path/to/wardens_icon.png", cv2.IMREAD_COLOR)
            assert service.wardens_icon is not None

    def test_init_logs_warning_when_colonial_icon_file_not_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that a warning is logged when colonial icon file is not found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = "/nonexistent/colonial.png"
        mock_settings.ocr.wardens_icon_path = None

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

    def test_init_logs_warning_when_wardens_icon_file_not_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that a warning is logged when wardens icon file is not found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        mock_settings.ocr.colonial_icon_path = None
        mock_settings.ocr.wardens_icon_path = "/nonexistent/wardens.png"

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=None,
        ):
            with patch(
                "verification_ocr.services.verification_service.logger.warning",
            ) as mock_warning:
                service = VerificationService(mock_settings)
                mock_warning.assert_called_once()
                assert "Failed to load wardens icon" in mock_warning.call_args[0][0]
                assert service.wardens_icon is None

    def test_init_colonial_icon_none_when_no_path(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that colonial icon is None when no path is provided.

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
        """
        Test that debug directory is created when debug mode is enabled.

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
        """
        Test that empty string is returned for None image.

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
        """
        Test that empty string is returned for empty image.

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
        """
        Test text extraction from valid image.

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
        """
        Test text extraction with scaling enabled.

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
        """
        Test text extraction without color inversion.

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
    """Tests for _detect_faction method."""

    def test_detect_faction_returns_none_when_no_templates(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """
        Test that None is returned when no faction icon templates are loaded.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        service = VerificationService(mock_settings)
        result = service._detect_faction(img)
        assert result is None

    def test_detect_faction_returns_colonial_when_colonial_matches(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """
        Test that COLONIAL is returned when colonial icon matches above threshold.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        service = VerificationService(mock_settings)
        # Set a mock colonial icon that will match
        service.colonial_icon = img.copy()

        result = service._detect_faction(img)
        assert result == Faction.COLONIAL

    def test_detect_faction_returns_wardens_when_wardens_matches(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """
        Test that WARDENS is returned when wardens icon matches above threshold.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
        assert img is not None

        service = VerificationService(mock_settings)
        # Set a mock wardens icon that will match
        service.wardens_icon = img.copy()

        result = service._detect_faction(img)
        assert result == Faction.WARDENS

    def test_find_user_info_sets_faction_colonial(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that faction is set to COLONIAL when colonial icon is found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        with patch.object(service, "_extract_text_from_image", return_value="TestPlayer"):
            with patch.object(service, "_detect_faction", return_value=Faction.COLONIAL):
                result = service._find_user_info(img, regions)

                assert result.faction == Faction.COLONIAL


class TestVerificationServiceCalculateRegions:
    """Tests for _calculate_regions method."""

    def test_calculate_regions_returns_correct_attributes(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that _calculate_regions returns ImageRegions with all expected attributes.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that scale factor is 1.0 for 4K resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        assert regions.scale_factor == 1.0

    def test_calculate_regions_scale_factor_for_1080p(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that scale factor is 0.5 for 1080p resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
        """
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        assert regions.scale_factor == 0.5


class TestVerificationServiceSaveDebugImage:
    """Tests for _save_debug_image method."""

    def test_save_debug_image_creates_file(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """
        Test that debug image is saved to file.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that invalid filenames are rejected.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that path components are stripped from filename.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that symlink-based path traversal is rejected.

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
        regions = service._calculate_regions(img)

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
        """
        Test that None is returned when image is None.

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
        """
        Test that empty image is returned unchanged.

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
        """
        Test that valid image is processed correctly.

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
        """
        Test that None is returned when shard region is empty.

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
        """
        Test that shard and time are extracted.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that None is returned when no text extracted.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test handling of single line OCR output.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

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
        """
        Test that ValueError is raised for invalid images.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            invalid_image_bytes (bytes): Invalid image bytes fixture.

        """
        service = VerificationService(mock_settings)
        with pytest.raises(ValueError, match="Failed to decode"):
            service.verify(invalid_image_bytes, invalid_image_bytes)

    def test_verify_raises_error_when_no_faction_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that ValueError is raised when no faction icon is found in either image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)
        # No faction icons loaded, so detection will return None
        with pytest.raises(ValueError, match="Could not detect faction icon in either image"):
            service.verify(image_bytes, image_bytes)

    def test_verify_raises_error_when_both_images_have_faction(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that ValueError is raised when both images contain faction icons.

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
        # Mock faction detection to return faction for both images
        with patch.object(service, "_detect_faction", return_value=Faction.COLONIAL):
            with pytest.raises(ValueError, match="Both images contain faction icons"):
                service.verify(image1_bytes, image2_bytes)

    def test_verify_raises_error_when_no_name_in_profile(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that ValueError is raised when no name is found in the profile image.

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
            # Mock: first image has faction (profile), second doesn't (shard)
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                with pytest.raises(ValueError, match="No player name found in the profile image"):
                    service.verify(profile_bytes, shard_bytes)

    def test_verify_extracts_user_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify extracts user information correctly.

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
            # Mock: first image has faction (profile), second doesn't (shard)
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.name == "TestPlayer"
                assert result.level == 25
                assert result.faction == Faction.COLONIAL

    def test_verify_handles_malformed_ocr_text(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify handles malformed OCR text gracefully.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.name == "TestPlayer"
                assert result.level is None

    def test_verify_handles_level_with_colon_but_no_digits(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that level is None when colon exists but no digits after it.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.name == "TestPlayer"
                assert result.level is None

    def test_verify_logs_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify logs an info message.

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
        """
        Test that user info is extracted from the image with faction icon.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                # Second image (profile) has faction
                return Faction.COLONIAL if faction_call_count[0] == 2 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(shard_bytes, profile_bytes)

                assert result.name == "SecondPlayer"
                assert result.shard == "ABLE"

    def test_verify_saves_debug_images_when_debug_mode(
        self,
        mock_settings: AppSettings,
        tmp_path: pathlib.Path,
    ) -> None:
        """
        Test that debug images are saved when debug mode is enabled.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                service.verify(profile_bytes, shard_bytes)

                assert os.path.exists(tmp_path / "debug_image1_regions.png")
                assert os.path.exists(tmp_path / "debug_image2_regions.png")

    def test_verify_includes_shard_and_ingame_time(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify includes shard and ingame_time in result.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.shard == "ABLE"
                assert result.ingame_time == "1234, 15:30"

    def test_verify_includes_war_number_and_current_ingame_time(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify includes war_number and current_ingame_time in result.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                result = service.verify(profile_bytes, shard_bytes)

                assert result.war_number == 132
                assert result.current_ingame_time is not None

    def test_verify_raises_error_when_no_shard_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that error is raised when no shard info is found.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
                with pytest.raises(
                    ValueError, match="No shard information found in the map/shard image"
                ):
                    service.verify(profile_bytes, shard_bytes)

    def test_verify_returns_result_when_war_not_configured(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify returns result when war state not configured.

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
            faction_call_count = [0]

            def mock_faction(img: Any) -> Faction | None:
                faction_call_count[0] += 1
                return Faction.COLONIAL if faction_call_count[0] == 1 else None

            with patch.object(service, "_detect_faction", side_effect=mock_faction):
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
        """
        Test verification with real colonial game screenshots.

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
        """
        Test verification with real warden game screenshots.

        Args:
            integration_settings (AppSettings): Integration settings with real icons.
            warden_image_bytes (bytes): Real warden user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        """
        service = VerificationService(integration_settings)
        result = service.verify(warden_image_bytes, stockpile_image_bytes)

        assert result.name is not None
        assert result.level is not None
        assert result.faction == Faction.WARDENS
        # Warden image has no regiment
        assert result.regiment is None

    def test_verify_extracts_shard_from_real_image(
        self,
        integration_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """
        Test shard extraction from real stockpile screenshot.

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
        """
        Test regiment name extraction from colonial user screenshot.

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
        """
        Test that warden user without regiment has None for regiment.

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
        """
        Test that ValueError is raised when image is too small.

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
        """
        Test that OpenCV errors are handled gracefully.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)

        with patch.object(
            service, "_calculate_regions", side_effect=cv2.error("Test OpenCV error")
        ):
            with pytest.raises(RuntimeError, match="Image processing error"):
                service.verify(image_bytes, image_bytes)

    def test_verify_handles_tesseract_error(
        self,
        integration_settings: AppSettings,
    ) -> None:
        """
        Test that Tesseract errors are handled gracefully.

        Args:
            integration_settings (AppSettings): Integration settings fixture.

        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(integration_settings)

        # First mock faction detection to work, then fail on _find_user_info
        faction_call_count = [0]

        def mock_faction(img: Any) -> Faction | None:
            faction_call_count[0] += 1
            return Faction.COLONIAL if faction_call_count[0] == 1 else None

        with patch.object(service, "_detect_faction", side_effect=mock_faction):
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
        """
        Test that unexpected exceptions are handled gracefully.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        service = VerificationService(mock_settings)

        with patch.object(service, "_calculate_regions", side_effect=Exception("Unexpected error")):
            with pytest.raises(RuntimeError, match="Internal processing error"):
                service.verify(image_bytes, image_bytes)
