"""Tests for verification service."""

import os
from unittest.mock import patch

import cv2
import numpy as np

from verification_ocr.core.settings import AppSettings
from verification_ocr.services.verification_service import (
    VerificationService,
    calculate_ingame_time_diff,
    extract_day_and_hour,
    get_current_ingame_time,
    parse_ingame_time,
)


class TestExtractDayAndHour:
    """Tests for extract_day_and_hour function."""

    def test_extracts_day_and_hour_with_two_commas(self) -> None:
        """
        Test extraction with two commas in input.

        Returns:
            None
        """
        result = extract_day_and_hour("Day 1,234, 15:30")
        assert result == "1234, 15:30"

    def test_extracts_day_and_hour_standard_format(self) -> None:
        """
        Test extraction with standard format.

        Returns:
            None
        """
        result = extract_day_and_hour("1234, 1530")
        assert result == "1234, 15:30"

    def test_returns_raw_result_when_no_time(self) -> None:
        """
        Test that raw result is returned when time format not found.

        Returns:
            None
        """
        result = extract_day_and_hour("12345")
        assert result == "12345"

    def test_returns_raw_result_when_digits_not_four(self) -> None:
        """
        Test that raw result is returned when right side has != 4 digits.

        Covers the branch when len(parts) == 2 but len(digits) != 4.

        Returns:
            None
        """
        # Has comma so len(parts) == 2, but "123" has only 3 digits
        result = extract_day_and_hour("100, 123")
        assert result == "100,123"

    def test_returns_formatted_when_extra_digits(self) -> None:
        """
        Test when right side has more than 4 digits.

        Returns:
            None
        """
        # "12345" has 5 digits, not 4
        result = extract_day_and_hour("100, 12345")
        assert result == "100,12345"

    def test_handles_empty_string(self) -> None:
        """
        Test handling of empty string.

        Returns:
            None
        """
        result = extract_day_and_hour("")
        assert result == ""


class TestParseIngameTime:
    """Tests for parse_ingame_time function."""

    def test_parses_valid_time(self) -> None:
        """
        Test parsing valid time string.

        Returns:
            None
        """
        result = parse_ingame_time("267, 21:45")
        assert result == (267, 21, 45)

    def test_returns_none_for_invalid_format(self) -> None:
        """
        Test returns None for invalid format.

        Returns:
            None
        """
        result = parse_ingame_time("invalid")
        assert result is None

    def test_returns_none_for_missing_time(self) -> None:
        """
        Test returns None when time part is missing.

        Returns:
            None
        """
        result = parse_ingame_time("267")
        assert result is None

    def test_returns_none_for_invalid_time_format(self) -> None:
        """
        Test returns None for invalid time format.

        Returns:
            None
        """
        result = parse_ingame_time("267, 2145")
        assert result is None

    def test_returns_none_for_non_numeric_values(self) -> None:
        """
        Test returns None for non-numeric values.

        Returns:
            None
        """
        result = parse_ingame_time("abc, de:fg")
        assert result is None


class TestCalculateIngameTimeDiff:
    """Tests for calculate_ingame_time_diff function."""

    def test_same_time_returns_zero(self) -> None:
        """
        Test that same times return zero difference.

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
        """
        result = get_current_ingame_time()
        assert result is None

    def test_returns_time_when_configured(self) -> None:
        """
        Test returns time when war state is configured.

        Returns:
            None
        """
        import time

        from verification_ocr.services.war_service import get_war_state

        state = get_war_state()
        # Set start time to 2.5 hours ago
        state.start_time = int((time.time() - 2.5 * 60 * 60) * 1000)

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

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        assert service.settings is mock_settings

    def test_init_sets_tesseract_cmd_when_provided(self) -> None:
        """
        Test that tesseract_cmd is set when provided in settings.

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
        """
        mock_settings.ocr.colonial_icon_path = "/path/to/icon.png"
        mock_icon = np.ones((50, 50, 3), dtype=np.uint8)

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=mock_icon,
        ) as mock_imread:
            service = VerificationService(mock_settings)
            mock_imread.assert_called_once_with("/path/to/icon.png", cv2.IMREAD_COLOR)
            assert service.colonial_icon is not None

    def test_init_logs_warning_when_colonial_icon_file_not_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that a warning is logged when colonial icon file is not found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        mock_settings.ocr.colonial_icon_path = "/nonexistent/icon.png"

        with patch(
            "verification_ocr.services.verification_service.cv2.imread",
            return_value=None,
        ):
            with patch(
                "verification_ocr.services.verification_service.logger.warning",
            ) as mock_warning:
                service = VerificationService(mock_settings)
                mock_warning.assert_called_once()
                assert "Failed to load" in mock_warning.call_args[0][0]
                assert service.colonial_icon is None

    def test_init_colonial_icon_none_when_no_path(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that colonial icon is None when no path is provided.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service._extract_text_from_image(None)
        assert result == ""

    def test_extract_text_returns_empty_for_empty_image(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that empty string is returned for empty image.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
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

        Returns:
            None
        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

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

        Returns:
            None
        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

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

        Returns:
            None
        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="ABLE\n",
        ):
            service = VerificationService(mock_settings)
            result = service._extract_text_from_image(image=img, use_invert=False)
            assert result == "ABLE"


class TestVerificationServiceFindColonialIcon:
    """Tests for _find_colonial_icon method."""

    def test_find_colonial_icon_returns_none_when_no_template(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """
        Test that None is returned when no colonial icon template.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        Returns:
            None
        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

        service = VerificationService(mock_settings)
        result = service._find_colonial_icon(img)
        assert result is None

    def test_find_colonial_icon_with_template(
        self,
        mock_settings: AppSettings,
        sample_image_bytes: bytes,
    ) -> None:
        """
        Test colonial icon detection with template.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            sample_image_bytes (bytes): Sample image bytes fixture.

        Returns:
            None
        """
        nparr = np.frombuffer(buffer=sample_image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

        service = VerificationService(mock_settings)
        # Set a mock colonial icon
        service.colonial_icon = np.ones((5, 5, 3), dtype=np.uint8)

        result = service._find_colonial_icon(img)
        assert isinstance(result, bool)


class TestVerificationServiceCalculateRegions:
    """Tests for _calculate_regions method."""

    def test_calculate_regions_returns_correct_keys(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that _calculate_regions returns all expected keys.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        assert "username" in regions
        assert "regiment" in regions
        assert "shard" in regions
        assert "scale_factor" in regions

    def test_calculate_regions_scale_factor_is_1_for_4k(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that scale factor is 1.0 for 4K resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        assert regions["scale_factor"] == 1.0

    def test_calculate_regions_scale_factor_for_1080p(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that scale factor is 0.5 for 1080p resolution.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        assert regions["scale_factor"] == 0.5


class TestVerificationServiceSaveDebugImage:
    """Tests for _save_debug_image method."""

    def test_save_debug_image_creates_file(
        self,
        mock_settings: AppSettings,
        tmp_path,
    ) -> None:
        """
        Test that debug image is saved to file.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        Returns:
            None
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

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service._prepare_image_for_shard_detection(image=None, scale_factor=1.0)
        assert result is None

    def test_returns_image_unchanged_when_empty(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that empty image is returned unchanged.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
        """
        # Create a very small image where shard region calculation results in empty slice
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        service = VerificationService(mock_settings)
        regions = service._calculate_regions(img)

        # Force empty shard region
        regions["shard"]["y1"] = 5
        regions["shard"]["y2"] = 5
        regions["shard"]["x1"] = 5
        regions["shard"]["x2"] = 5

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

        Returns:
            None
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

        Returns:
            None
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

        Returns:
            None
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

    def test_verify_returns_error_for_invalid_images(
        self,
        mock_settings: AppSettings,
        invalid_image_bytes: bytes,
    ) -> None:
        """
        Test that error is returned for invalid images.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            invalid_image_bytes (bytes): Invalid image bytes fixture.

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service.verify(invalid_image_bytes, invalid_image_bytes)

        assert result["success"] is False
        assert "Failed to decode" in result["error"]

    def test_verify_returns_error_when_no_name_found(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that error is returned when no name is found.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        # Create a larger test image (small images cause empty regions)
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="",
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            assert result["success"] is False
            assert "No name found" in result["error"]

    def test_verify_extracts_user_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify extracts user information correctly.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        # Create a larger test image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="TestPlayer Icon Level: 25\n",
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            assert result["success"] is True
            assert result["verification"]["name"] == "TestPlayer"
            assert result["verification"]["level"] == 25

    def test_verify_handles_malformed_ocr_text(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify handles malformed OCR text gracefully.

        Tests the except (IndexError, ValueError) branch in _find_user_info.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        # "Level: abc" will cause ValueError when trying int(parts[1])
        # Name is still extracted, but level will be None
        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="TestPlayer Icon Level: abc\n",
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            # Should not crash - name extracted but level parsing failed gracefully
            assert result["success"] is True
            assert result["verification"]["name"] == "TestPlayer"
            assert result["verification"]["level"] is None

    def test_verify_logs_info(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify logs an info message.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
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
                service.verify(image_bytes, image_bytes)

                mock_log.assert_called_once_with("Processing image pair for verification")

    def test_verify_tries_second_image_if_first_has_no_name(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that second image is tried if first has no name.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            # Call 1: first image username -> empty (no name found, returns early)
            # Call 2: second image username -> return name
            # Call 3: second image regiment -> doesn't matter
            # Call 4: first image shard -> doesn't matter
            if call_count[0] == 1:
                return ""
            if call_count[0] == 2:
                return "SecondPlayer Icon Level: 30\n"
            return "ABLE\n"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            assert result["success"] is True
            assert result["verification"]["name"] == "SecondPlayer"

    def test_verify_saves_debug_images_when_debug_mode(
        self,
        mock_settings: AppSettings,
        tmp_path,
    ) -> None:
        """
        Test that debug images are saved when debug mode is enabled.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            tmp_path: Pytest temporary path fixture.

        Returns:
            None
        """
        mock_settings.ocr.debug_mode = True
        mock_settings.ocr.debug_output_dir = str(tmp_path)

        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="TestPlayer Icon Level: 25\n",
        ):
            service = VerificationService(mock_settings)
            service.verify(image_bytes, image_bytes)

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

        Returns:
            None
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            # Call 1: username extraction for image1
            # Call 2: regiment extraction for image1
            # Call 3: shard/time extraction for image2
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""  # Regiment
            return "1234, 1530\nABLE"  # Shard and time

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            assert result["success"] is True
            assert result["verification"]["shard"] == "ABLE"
            assert result["verification"]["ingame_time"] == "1234, 15:30"

    def test_verify_fails_when_time_diff_exceeds_max(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify fails when in-game time diff exceeds max.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.core.settings.app_settings import VerificationSettings
        from verification_ocr.services.war_service import get_war_state

        # Set up war state with current time
        state = get_war_state()
        state.start_time = int((time.time() - 100 * 60 * 60) * 1000)  # 100 hours ago

        # Set strict max time diff
        mock_settings.verification = VerificationSettings(max_ingame_time_diff=10)

        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""
            # Return a time that's way off from current (day 1 instead of day 101)
            return "1, 0000\nABLE"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            assert result["success"] is False
            assert "In-game time difference" in result["error"]
            assert result["verification"] is not None

    def test_verify_skips_time_validation_when_ingame_time_none(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that time validation is skipped when ingame_time is None.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""
            # Return empty string so ingame_time will be None
            return ""

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            # Should succeed since time validation is skipped
            assert result["success"] is True
            assert result["verification"]["ingame_time"] is None

    def test_verify_skips_time_validation_when_time_cannot_be_parsed(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that time validation is skipped when ingame_time cannot be parsed.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.services.war_service import get_war_state

        # Set up war state so current_time is available
        state = get_war_state()
        state.start_time = int((time.time() - 100 * 60 * 60) * 1000)

        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""
            # Return invalid time format that cannot be parsed
            return "invalid_time\nABLE"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            # Should succeed since parsed_time is None
            assert result["success"] is True

    def test_verify_skips_time_validation_when_war_not_configured(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that time validation is skipped when war state not configured.

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        # War state is reset by conftest, so current_time will be None

        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""
            # Return valid time format
            return "100, 1200\nABLE"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            # Should succeed since current_time is None (war not configured)
            assert result["success"] is True
            assert result["verification"]["ingame_time"] == "100, 12:00"

    def test_verify_succeeds_when_time_diff_within_max(
        self,
        mock_settings: AppSettings,
    ) -> None:
        """
        Test that verify succeeds when time diff is within max allowed.

        Covers the branch where time validation passes (time_diff <= max_diff).

        Args:
            mock_settings (AppSettings): Mock settings fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.core.settings.app_settings import VerificationSettings
        from verification_ocr.services.war_service import get_war_state

        # Set up war state - 100 hours ago means we're on day 101
        state = get_war_state()
        state.start_time = int((time.time() - 100 * 60 * 60) * 1000)

        # Set max time diff to 50 hours (plenty of room)
        mock_settings.verification = VerificationSettings(max_ingame_time_diff=50)

        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        image_bytes = buffer.tobytes()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestPlayer Icon Level: 25\n"
            if call_count[0] == 2:
                return ""
            # Return time close to current (day 101, hour 0)
            return "101, 0000\nABLE"

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            service = VerificationService(mock_settings)
            result = service.verify(image_bytes, image_bytes)

            # Should succeed since time diff is within max
            assert result["success"] is True
            assert result["verification"]["ingame_time"] == "101, 00:00"


class TestVerificationServiceIntegration:
    """Integration tests using real game screenshots."""

    def test_verify_colonial_with_real_images(
        self,
        mock_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """
        Test verification with real colonial game screenshots.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            colonial_image_bytes (bytes): Real colonial user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service.verify(colonial_image_bytes, stockpile_image_bytes)

        assert result["success"] is True
        assert result["verification"]["name"] is not None
        assert result["verification"]["level"] is not None

    def test_verify_warden_with_real_images(
        self,
        mock_settings: AppSettings,
        warden_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """
        Test verification with real warden game screenshots.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            warden_image_bytes (bytes): Real warden user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service.verify(warden_image_bytes, stockpile_image_bytes)

        assert result["success"] is True
        assert result["verification"]["name"] is not None
        assert result["verification"]["level"] is not None
        # Warden should not be detected as colonial (False or None if template not configured)
        assert result["verification"]["colonial"] is not True
        # Warden image has no regiment (False or None)
        assert result["verification"]["regiment"] is not True

    def test_verify_extracts_shard_from_real_image(
        self,
        mock_settings: AppSettings,
        colonial_image_bytes: bytes,
        stockpile_image_bytes: bytes,
    ) -> None:
        """
        Test shard extraction from real stockpile screenshot.

        Args:
            mock_settings (AppSettings): Mock settings fixture.
            colonial_image_bytes (bytes): Real colonial user screenshot.
            stockpile_image_bytes (bytes): Real stockpile screenshot.

        Returns:
            None
        """
        service = VerificationService(mock_settings)
        result = service.verify(colonial_image_bytes, stockpile_image_bytes)

        assert result["success"] is True
        # Shard should be extracted from stockpile image
        assert (
            result["verification"]["shard"] is not None
            or result["verification"]["ingame_time"] is not None
        )
