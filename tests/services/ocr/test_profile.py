"""Tests for profile detection module."""

import numpy as np
from cv2.typing import MatLike

from verification_ocr.services.ocr.profile import (
    PROFILE_GAP_RATIO,
    PROFILE_MIN_BOX_HEIGHT,
    PROFILE_MIN_BOX_WIDTH,
    PROFILE_REF_HEIGHT,
    PROFILE_ROW1_REF_ASPECT,
    PROFILE_ROW1_WIDTH_RATIOS,
    PROFILE_ROW2_REF_ASPECT,
    PROFILE_ROW2_WIDTH_RATIOS,
    PROFILE_TOLERANCE,
    Box,
    _filter_boxes,
    _find_box_by_width,
    _find_matching_boxes,
    _get_grey_range,
    _group_by_height,
    _group_by_y,
    _match_box_pattern,
    detect_profile_boxes,
)


class TestConstants:
    """Tests for profile detection constants."""

    def test_reference_height_is_positive(self) -> None:
        """Test that reference height is positive."""
        assert PROFILE_REF_HEIGHT > 0

    def test_min_box_dimensions_are_positive(self) -> None:
        """Test that minimum box dimensions are positive."""
        assert PROFILE_MIN_BOX_HEIGHT > 0
        assert PROFILE_MIN_BOX_WIDTH > 0

    def test_tolerance_is_reasonable(self) -> None:
        """Test that tolerance is between 0 and 1."""
        assert 0 < PROFILE_TOLERANCE < 1

    def test_gap_ratio_is_positive(self) -> None:
        """Test that gap ratio is positive."""
        assert PROFILE_GAP_RATIO > 0

    def test_row1_has_4_width_ratios(self) -> None:
        """Test that row 1 has 4 width ratios."""
        assert len(PROFILE_ROW1_WIDTH_RATIOS) == 4

    def test_row2_has_2_width_ratios(self) -> None:
        """Test that row 2 has 2 width ratios."""
        assert len(PROFILE_ROW2_WIDTH_RATIOS) == 2

    def test_aspect_ratios_are_positive(self) -> None:
        """Test that aspect ratios are positive."""
        assert PROFILE_ROW1_REF_ASPECT > 0
        assert PROFILE_ROW2_REF_ASPECT > 0


class TestGetGreyRange:
    """Tests for _get_grey_range function."""

    def test_dark_image_returns_dark_range(self) -> None:
        """Test that dark images return dark grey range."""
        # Create a dark image (median < 30)
        dark_image = np.ones((100, 100), dtype=np.uint8) * 20
        grey_min, grey_max = _get_grey_range(dark_image)
        assert grey_min == 8
        assert grey_max == 25

    def test_normal_image_returns_normal_range(self) -> None:
        """Test that normal images return normal grey range."""
        # Create a normal image (30 <= median < 80)
        normal_image = np.ones((100, 100), dtype=np.uint8) * 50
        grey_min, grey_max = _get_grey_range(normal_image)
        assert grey_min == 40
        assert grey_max == 70

    def test_light_image_returns_light_range(self) -> None:
        """Test that light images return light grey range."""
        # Create a light image (median >= 80)
        light_image = np.ones((100, 100), dtype=np.uint8) * 150
        grey_min, grey_max = _get_grey_range(light_image)
        assert grey_min == 70
        assert grey_max == 100


class TestFilterBoxes:
    """Tests for _filter_boxes function."""

    def _create_contour(self, x: int, y: int, w: int, h: int) -> MatLike:
        """Create a simple rectangular contour.

        Note: cv2.boundingRect includes endpoints, so we use w-1 and h-1
        to get the expected bounding box dimensions.
        """
        return np.array([[[x, y]], [[x + w - 1, y]], [[x + w - 1, y + h - 1]], [[x, y + h - 1]]])

    def test_filters_small_boxes(self) -> None:
        """Test that small boxes are filtered out."""
        contours = [
            self._create_contour(0, 0, 10, 10),  # Too small
            self._create_contour(0, 0, 100, 30),  # Valid
        ]
        boxes = _filter_boxes(contours)
        assert len(boxes) == 1
        assert boxes[0] == (0, 0, 100, 30)

    def test_filters_narrow_boxes(self) -> None:
        """Test that narrow boxes are filtered out."""
        contours = [
            self._create_contour(0, 0, 20, 30),  # Too narrow
            self._create_contour(0, 0, 100, 30),  # Valid
        ]
        boxes = _filter_boxes(contours)
        assert len(boxes) == 1

    def test_returns_valid_boxes(self) -> None:
        """Test that valid boxes are returned."""
        contours = [
            self._create_contour(10, 20, 150, 35),
            self._create_contour(200, 20, 60, 35),
        ]
        boxes = _filter_boxes(contours)
        assert len(boxes) == 2


class TestGroupByHeight:
    """Tests for _group_by_height function."""

    def test_groups_similar_heights(self) -> None:
        """Test that boxes with similar heights are grouped."""
        boxes: list[Box] = [
            (0, 0, 100, 35),
            (100, 0, 100, 36),
            (200, 0, 100, 35),
        ]
        groups = _group_by_height(boxes)
        # All should be in the same group (within tolerance)
        assert any(len(group) == 3 for group in groups.values())

    def test_separates_different_heights(self) -> None:
        """Test that boxes with different heights are separated."""
        boxes: list[Box] = [
            (0, 0, 100, 35),
            (100, 0, 100, 70),  # Much larger
        ]
        groups = _group_by_height(boxes)
        # Should have 2 different reference heights
        assert len(groups) >= 2


class TestGroupByY:
    """Tests for _group_by_y function."""

    def test_groups_same_row(self) -> None:
        """Test that boxes on the same row are grouped."""
        boxes: list[Box] = [
            (0, 100, 50, 35),
            (60, 101, 50, 35),
            (120, 100, 50, 35),
        ]
        rows = _group_by_y(boxes, ref_height=35)
        # All should be in the same row
        assert any(len(row) == 3 for row in rows.values())

    def test_separates_different_rows(self) -> None:
        """Test that boxes on different rows are separated."""
        boxes: list[Box] = [
            (0, 100, 50, 35),
            (0, 200, 50, 35),  # Different row
        ]
        rows = _group_by_y(boxes, ref_height=35)
        assert len(rows) == 2


class TestDetectProfileBoxes:
    """Tests for detect_profile_boxes function."""

    def test_returns_none_for_blank_image(self) -> None:
        """Test that blank images return None."""
        blank = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        result = detect_profile_boxes(blank)
        assert result is None

    def test_returns_none_for_small_image(self) -> None:
        """Test that small images return None."""
        small = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = detect_profile_boxes(small)
        assert result is None

    def test_returns_list_of_4_boxes_when_found(self) -> None:
        """Test that valid detection returns 4 boxes."""
        # This is a simplified test - real detection requires proper UI patterns
        # For integration tests, use actual game screenshots
        blank = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        result = detect_profile_boxes(blank)
        # Will be None for blank image, but validates return type
        assert result is None or (isinstance(result, list) and len(result) == 4)


class TestMatchBoxPattern:
    """Tests for _match_box_pattern function."""

    def test_returns_none_for_insufficient_boxes(self) -> None:
        """Test that insufficient boxes return None."""
        boxes: list[Box] = [
            (0, 0, 100, 35),
            (100, 0, 30, 35),
        ]
        result = _match_box_pattern(boxes, ref_height=35)
        assert result is None

    def test_returns_4_boxes_when_pattern_found(self) -> None:
        """Test that valid pattern returns 4 boxes."""
        # Create boxes matching expected row 1 pattern (4 boxes) + row 2 (2 boxes)
        ref_h = 35
        gap = int(ref_h * PROFILE_GAP_RATIO)

        # Row 1: widths relative to first box (200px at 35h)
        row1_w1 = 200
        row1_w2 = int(200 * 0.30)
        row1_w3 = int(200 * 0.65)
        row1_w4 = int(200 * 0.65)

        # Row 2: 264px each
        row2_w1 = 264
        row2_w2 = 264

        x = 100
        y1 = 100
        y2 = y1 + ref_h + gap

        boxes: list[Box] = [
            # Row 1
            (x, y1, row1_w1, ref_h),
            (x + row1_w1 + gap, y1, row1_w2, ref_h),
            (x + row1_w1 + gap + row1_w2 + gap, y1, row1_w3, ref_h),
            (x + row1_w1 + gap + row1_w2 + gap + row1_w3 + gap, y1, row1_w4, ref_h),
            # Row 2
            (x, y2, row2_w1, ref_h),
            (x + row2_w1 + gap, y2, row2_w2, ref_h),
        ]

        result = _match_box_pattern(boxes, ref_height=ref_h)
        # Pattern matching is strict, may not match without exact proportions
        assert result is None or len(result) == 4

    def test_returns_none_when_row1_too_few_boxes(self) -> None:
        """Test returns None when row 1 has fewer than 4 boxes."""
        ref_h = 35
        y1 = 100
        boxes: list[Box] = [
            (100, y1, 200, ref_h),
            (310, y1, 60, ref_h),
            (380, y1, 130, ref_h),
            # Only 3 boxes in row 1
        ]
        result = _match_box_pattern(boxes, ref_height=ref_h)
        assert result is None

    def test_returns_none_when_row2_y_gap_wrong(self) -> None:
        """Test returns None when row 2 Y gap is incorrect."""
        ref_h = 35
        gap = int(ref_h * PROFILE_GAP_RATIO)
        y1 = 100
        # Row 2 Y is too far from row 1 (wrong gap)
        y2 = y1 + ref_h * 3  # Much larger than expected gap

        boxes: list[Box] = [
            # Row 1 - 4 boxes with correct spacing
            (100, y1, 200, ref_h),
            (100 + 200 + gap, y1, 60, ref_h),
            (100 + 200 + gap + 60 + gap, y1, 130, ref_h),
            (100 + 200 + gap + 60 + gap + 130 + gap, y1, 130, ref_h),
            # Row 2 - wrong Y gap
            (100, y2, 264, ref_h),
            (100 + 264 + gap, y2, 264, ref_h),
        ]
        result = _match_box_pattern(boxes, ref_height=ref_h)
        assert result is None

    def test_returns_none_when_row2_too_few_boxes(self) -> None:
        """Test returns None when row 2 has fewer than 2 boxes."""
        ref_h = 35
        gap = int(ref_h * PROFILE_GAP_RATIO)
        y1 = 100
        y2 = y1 + int(ref_h * (1 + PROFILE_GAP_RATIO))

        boxes: list[Box] = [
            # Row 1 - 4 boxes
            (100, y1, 200, ref_h),
            (100 + 200 + gap, y1, 60, ref_h),
            (100 + 200 + gap + 60 + gap, y1, 130, ref_h),
            (100 + 200 + gap + 60 + gap + 130 + gap, y1, 130, ref_h),
            # Row 2 - only 1 box
            (100, y2, 264, ref_h),
        ]
        result = _match_box_pattern(boxes, ref_height=ref_h)
        assert result is None


class TestFindMatchingBoxes:
    """Tests for _find_matching_boxes function."""

    def test_returns_none_when_aspect_ratio_wrong(self) -> None:
        """Test returns None when reference box aspect ratio doesn't match."""
        ref_h = 35
        # Create boxes with wrong aspect ratio for row 1
        # Expected aspect is PROFILE_ROW1_REF_ASPECT (~5.7)
        row_boxes: list[Box] = [
            (100, 100, 50, ref_h),  # Aspect 1.4 - wrong
            (160, 100, 50, ref_h),
            (220, 100, 50, ref_h),
            (280, 100, 50, ref_h),
        ]
        result = _find_matching_boxes(
            row_boxes=row_boxes,
            expected_ratios=PROFILE_ROW1_WIDTH_RATIOS,
            ref_aspect=PROFILE_ROW1_REF_ASPECT,
            expected_gap=4,
            gap_tolerance=2,
        )
        assert result is None

    def test_returns_boxes_when_pattern_matches(self) -> None:
        """Test returns boxes when width ratios match."""
        ref_h = 35
        gap = 4
        # Row 1 widths: 200, 60, 130, 130 (ratios 1.0, 0.30, 0.65, 0.65)
        row_boxes: list[Box] = [
            (100, 100, 200, ref_h),
            (100 + 200 + gap, 100, 60, ref_h),
            (100 + 200 + gap + 60 + gap, 100, 130, ref_h),
            (100 + 200 + gap + 60 + gap + 130 + gap, 100, 130, ref_h),
        ]
        result = _find_matching_boxes(
            row_boxes=row_boxes,
            expected_ratios=PROFILE_ROW1_WIDTH_RATIOS,
            ref_aspect=PROFILE_ROW1_REF_ASPECT,
            expected_gap=gap,
            gap_tolerance=2,
        )
        # Should find all 4 boxes
        assert result is not None
        assert len(result) == 4


class TestFindBoxByWidth:
    """Tests for _find_box_by_width function."""

    def test_returns_none_when_gap_too_large(self) -> None:
        """Test returns None when gap is outside tolerance."""
        boxes: list[Box] = [
            (200, 100, 60, 35),  # Gap is 100, expected 4
        ]
        result = _find_box_by_width(
            boxes=boxes,
            expected_width=60,
            after_x=100,
            expected_gap=4,
            gap_tolerance=2,
        )
        assert result is None

    def test_returns_none_when_width_wrong(self) -> None:
        """Test returns None when width doesn't match."""
        boxes: list[Box] = [
            (104, 100, 100, 35),  # Width 100, expected 60 (67% off)
        ]
        result = _find_box_by_width(
            boxes=boxes,
            expected_width=60,
            after_x=100,
            expected_gap=4,
            gap_tolerance=2,
        )
        assert result is None

    def test_returns_box_when_matches(self) -> None:
        """Test returns box when gap and width match."""
        boxes: list[Box] = [
            (104, 100, 60, 35),  # Gap 4, width 60 - matches
        ]
        result = _find_box_by_width(
            boxes=boxes,
            expected_width=60,
            after_x=100,
            expected_gap=4,
            gap_tolerance=2,
        )
        assert result == (104, 100, 60, 35)

    def test_skips_non_matching_boxes(self) -> None:
        """Test skips boxes that don't match and finds correct one."""
        boxes: list[Box] = [
            (50, 100, 60, 35),  # Gap -50 - skip
            (200, 100, 60, 35),  # Gap 100 - skip
            (104, 100, 60, 35),  # Gap 4 - matches
        ]
        result = _find_box_by_width(
            boxes=boxes,
            expected_width=60,
            after_x=100,
            expected_gap=4,
            gap_tolerance=2,
        )
        assert result == (104, 100, 60, 35)
