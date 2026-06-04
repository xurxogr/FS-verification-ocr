"""Tests for shard region detection module."""

import numpy as np

from verification_ocr.services.ocr.shard import (
    SHARD_CROP_HEIGHT_RATIO,
    SHARD_CROP_WIDTH_RATIO,
    SHARD_OFFSET_BOTTOM_RATIO,
    SHARD_OFFSET_X_RATIO,
    detect_shard_region,
)


class TestConstants:
    """Tests for shard detection constants."""

    def test_crop_ratios_are_positive(self) -> None:
        """Test that crop ratios are positive."""
        assert SHARD_CROP_WIDTH_RATIO > 0
        assert SHARD_CROP_HEIGHT_RATIO > 0


class TestDetectShardRegion:
    """Tests for detect_shard_region function."""

    def test_returns_box_for_valid_image(self) -> None:
        """Test that a box is returned for valid image dimensions."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = detect_shard_region(image=img, profile_height=35)

        assert result is not None
        x, y, w, h = result
        # Reference measurements at 1920x1080, profile_height=35
        assert x == 34
        assert y == 1080 - 112
        assert w == 150
        assert h == 48
        assert x == int(35 * SHARD_OFFSET_X_RATIO)
        assert y == 1080 - int(35 * SHARD_OFFSET_BOTTOM_RATIO)
        assert w == int(35 * SHARD_CROP_WIDTH_RATIO)
        assert h == int(35 * SHARD_CROP_HEIGHT_RATIO)

    def test_returns_none_for_small_image(self) -> None:
        """Test that None is returned for too-small images."""
        small_img = np.zeros((50, 50, 3), dtype=np.uint8)

        result = detect_shard_region(image=small_img, profile_height=35)

        assert result is None

    def test_box_y_is_near_bottom(self) -> None:
        """Test that box y coordinate is near bottom of image."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = detect_shard_region(image=img, profile_height=35)

        assert result is not None
        _, y, _, h = result
        # Block sits near (but not flush with) the bottom edge
        assert y > 1080 * 0.8
        assert y + h <= 1080

    def test_scales_with_profile_height(self) -> None:
        """Test that box scales with profile height."""
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)

        box_35 = detect_shard_region(image=img, profile_height=35)
        box_70 = detect_shard_region(image=img, profile_height=70)

        assert box_35 is not None
        assert box_70 is not None

        # 4K box should be ~2x the 1080p box dimensions (allow for int rounding)
        assert abs(box_70[2] - box_35[2] * 2) <= 1  # width
        assert abs(box_70[3] - box_35[3] * 2) <= 1  # height
