"""Profile detection for OCR service."""

import math
from collections.abc import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

# Box coordinates: (x, y, width, height)
type Box = tuple[int, int, int, int]

# Grey value ranges based on image median (threshold, (min, max))
# Dark theme (median < 30): grey boxes ~8-25
# Normal theme (30 <= median < 80): grey boxes ~40-70
# Light theme (median >= 80): grey boxes ~70-100
PROFILE_GREY_RANGES = [
    (30, (8, 25)),  # dark theme
    (80, (40, 70)),  # normal theme
    (256, (70, 100)),  # light theme (256 = always match)
]

# Reference height at 1080p (used for scaling calculations)
PROFILE_REF_HEIGHT = 35

# Grey box aspect ratios (width/height) for reference boxes
# Row 1 at 1080p: 200px / 35px = 5.71 (username is reference)
PROFILE_ROW1_REF_ASPECT = 200 / PROFILE_REF_HEIGHT
# Row 2 at 1080p: 264px / 35px = 7.54 (commends is reference)
PROFILE_ROW2_REF_ASPECT = 264 / PROFILE_REF_HEIGHT

# Grey box width ratios (relative to first box in each row)
# Row 1 at 1080p: 200px, 60px, 130px, 130px (username, icon, level, rank)
PROFILE_ROW1_WIDTH_RATIOS = [1.0, 0.30, 0.65, 0.65]
# Row 2 at 1080p: 264px, 264px (commends, communications)
PROFILE_ROW2_WIDTH_RATIOS = [1.0, 1.0]

# Tolerance for all ratio/proportion matching (5%)
PROFILE_TOLERANCE = 0.05

# Gap between boxes (4px at 35px height, used for X and Y gaps)
PROFILE_GAP_RATIO = 4 / PROFILE_REF_HEIGHT

# Minimum box dimensions (based on 768p, smallest supported)
PROFILE_MIN_BOX_HEIGHT = 25
PROFILE_MIN_BOX_WIDTH = 43


def detect_profile_boxes(image: MatLike) -> list[Box] | None:
    """Detect the 4 profile boxes needed for verification.

    Args:
        image: BGR image to analyze.

    Returns:
        List of 4 boxes (x, y, w, h): [Username, Icon, Level, Regiment],
        or None if not found.
    """
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Get detected range first, then try others
    detected_range = _get_grey_range(gray)
    all_ranges = [r[1] for r in PROFILE_GREY_RANGES]

    # Reorder: detected range first, then others
    ranges_to_try = [detected_range] + [r for r in all_ranges if r != detected_range]

    for grey_min, grey_max in ranges_to_try:
        result = _try_detect_with_range(gray, grey_min, grey_max)
        if result is not None:
            return result

    return None


def _try_detect_with_range(
    gray: MatLike,
    grey_min: int,
    grey_max: int,
) -> list[Box] | None:
    """Try to detect profile boxes using a specific grey range.

    Args:
        gray: Grayscale image.
        grey_min: Minimum grey value.
        grey_max: Maximum grey value.

    Returns:
        List of 4 boxes or None if not found.
    """
    # Convert pixels below grey range to white to fill text holes in grey boxes
    mask_dark = cv2.inRange(
        src=gray,
        lowerb=np.array([0]),
        upperb=np.array([grey_min - 1]),
    )
    gray_filled = gray.copy()
    gray_filled[mask_dark > 0] = 255

    grey_mask = cv2.inRange(
        src=gray_filled,
        lowerb=np.array([grey_min]),
        upperb=np.array([grey_max]),
    )

    contours, _ = cv2.findContours(
        image=grey_mask,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # Filter contours that don't meet minimum box width and height
    boxes = _filter_boxes(contours)
    if len(boxes) < 6:
        return None

    # Group boxes by similar height to find candidates for the 4+2 row pattern
    height_groups = _group_by_height(boxes)

    for ref_height, group in height_groups.items():
        if len(group) < 6:
            continue
        result = _match_box_pattern(boxes=group, ref_height=ref_height)
        if result is not None:
            return result

    return None


def _get_grey_range(gray: MatLike) -> tuple[int, int]:
    """Determine grey box value range based on image median.

    Args:
        gray: Grayscale image.

    Returns:
        Tuple of (min, max) grey values for the detected theme.
    """
    median = int(np.median(gray))

    for threshold, grey_range in PROFILE_GREY_RANGES:
        if median < threshold:
            return grey_range

    return PROFILE_GREY_RANGES[-1][1]


def _filter_boxes(
    contours: Sequence[MatLike],
) -> list[Box]:
    """Filter contours to valid box candidates.

    Args:
        contours: Contours from cv2.findContours.

    Returns:
        List of boxes (x, y, w, h) that meet minimum size requirements.
    """
    boxes: list[Box] = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        if h >= PROFILE_MIN_BOX_HEIGHT and w >= PROFILE_MIN_BOX_WIDTH:
            boxes.append((x, y, w, h))

    return boxes


def _group_by_height(
    boxes: list[Box],
) -> dict[int, list[Box]]:
    """Group boxes by similar height.

    First groups by exact height, then combines heights within tolerance.

    Args:
        boxes: List of boxes (x, y, w, h).

    Returns:
        Dictionary mapping reference height to list of boxes within tolerance.
    """
    # Step 1: Group by exact height
    exact_groups: dict[int, list[Box]] = {}
    for box in boxes:
        h = box[3]
        if h not in exact_groups:
            exact_groups[h] = []
        exact_groups[h].append(box)

    # Step 2: For each unique height, collect all boxes within tolerance
    combined_groups: dict[int, list[Box]] = {}
    for ref_h in sorted(exact_groups.keys()):
        group_boxes: list[Box] = []
        for h, boxes_at_h in exact_groups.items():
            if abs(h - ref_h) / ref_h <= PROFILE_TOLERANCE:
                group_boxes.extend(boxes_at_h)
        combined_groups[ref_h] = group_boxes

    return combined_groups


def _find_matching_boxes(
    row_boxes: list[Box],
    expected_ratios: list[float],
    ref_aspect: float,
    expected_gap: float,
    gap_tolerance: float,
) -> list[Box] | None:
    """Find boxes matching expected width ratios, skipping non-matching boxes.

    For each candidate reference box, validates its aspect ratio matches
    expected, then calculates expected widths for other boxes and finds
    matches. Non-matching boxes are ignored.

    Args:
        row_boxes: Boxes in a row, sorted by X position.
        expected_ratios: Expected width ratios relative to first box.
        ref_aspect: Expected aspect ratio (width/height) for reference box.
        expected_gap: Expected gap between adjacent boxes.
        gap_tolerance: Allowed deviation from expected gap (±tolerance).

    Returns:
        Matching boxes in X order, or None if no valid pattern found.
    """
    # Try each box as the reference (first) box
    for ref_idx, ref_box in enumerate(row_boxes):
        ref_width = ref_box[2]
        ref_height = ref_box[3]

        # Validate reference box has expected aspect ratio
        actual_aspect = ref_width / ref_height
        if abs(actual_aspect - ref_aspect) / ref_aspect > PROFILE_TOLERANCE:
            continue

        matched: list[Box] = [ref_box]

        # Calculate expected widths for remaining boxes
        expected_widths = [ref_width * ratio for ratio in expected_ratios[1:]]

        # Search remaining boxes (to the right) for matches
        remaining = row_boxes[ref_idx + 1 :]
        for expected_w in expected_widths:
            found = _find_box_by_width(
                boxes=remaining,
                expected_width=expected_w,
                after_x=matched[-1][0] + matched[-1][2],
                expected_gap=expected_gap,
                gap_tolerance=gap_tolerance,
            )
            if found is None:
                break
            matched.append(found)
            # Update remaining to only boxes after the found one
            found_idx = remaining.index(found)
            remaining = remaining[found_idx + 1 :]

        if len(matched) == len(expected_ratios):
            return matched

    return None


def _find_box_by_width(
    boxes: list[Box],
    expected_width: float,
    after_x: int,
    expected_gap: float,
    gap_tolerance: float,
) -> Box | None:
    """Find a box matching expected width within gap constraints.

    Args:
        boxes: Candidate boxes sorted by X position.
        expected_width: Expected box width (can be float from ratio calculation).
        after_x: X position after which to search (right edge of previous box).
        expected_gap: Expected gap from after_x to box start.
        gap_tolerance: Allowed deviation from expected gap (±tolerance).

    Returns:
        First matching box, or None if not found.
    """
    for box in boxes:
        x, _, w, _ = box
        gap = x - after_x

        # Skip if gap doesn't match expected ±tolerance
        if abs(gap - expected_gap) > gap_tolerance:
            continue

        # Check if width matches within tolerance
        if abs(w - expected_width) / expected_width <= PROFILE_TOLERANCE:
            return box

    return None


def _match_box_pattern(
    boxes: list[Box],
    ref_height: int,
) -> list[Box] | None:
    """Match boxes against the expected 4+2 row pattern and extract needed boxes.

    Profile UI layout:
        Row 1: [Username] [Icon] [Level] [Rank]  (4 boxes, width ratios 1.0, 0.30, 0.65, 0.65)
        Row 2: [Commends] [Communications]       (2 boxes, width ratios 1.0, 1.0)
        Regiment: calculated from row2 box2

    Args:
        boxes: List of boxes (x, y, w, h) with similar height.
        ref_height: Reference height for Y-position grouping.

    Returns:
        List of 4 boxes [Username, Icon, Level, Regiment] or None if not found.
    """
    # Group boxes by Y position to identify rows
    rows = _group_by_y(boxes=boxes, ref_height=ref_height)
    sorted_ys = sorted(rows.keys())

    # Find row 1: need 4+ boxes, find 4 consecutive that match width ratios
    for i, row1_y in enumerate(sorted_ys):
        row1_all = rows[row1_y]
        if len(row1_all) < 4:
            continue

        row1_sorted = sorted(row1_all, key=lambda b: b[0])
        actual_height = row1_sorted[0][3]

        # Calculate expected gap: 4px at 35px height, scaled proportionally
        expected_x_gap = actual_height * PROFILE_GAP_RATIO
        # Tolerance: 5% of expected gap, rounded up (minimum ±1px)
        gap_tolerance = math.ceil(expected_x_gap * PROFILE_TOLERANCE)

        # Try to find 4 boxes matching row 1 width ratios (skip non-matching)
        row1 = _find_matching_boxes(
            row_boxes=row1_sorted,
            expected_ratios=PROFILE_ROW1_WIDTH_RATIOS,
            ref_aspect=PROFILE_ROW1_REF_ASPECT,
            expected_gap=expected_x_gap,
            gap_tolerance=gap_tolerance,
        )
        if row1 is None:
            continue

        # Find row 2: must be below row 1 with correct Y gap (height + separator)
        # Expected Y gap = height + 4px separator at 35px = height * (1 + GAP_RATIO)
        expected_y_gap = actual_height * (1 + PROFILE_GAP_RATIO)

        for row2_y in sorted_ys[i + 1 :]:
            row2_all = rows[row2_y]
            if len(row2_all) < 2:
                continue

            # Check Y gap is within tolerance of expected
            row_gap = row2_y - row1_y
            if abs(row_gap - expected_y_gap) / expected_y_gap > PROFILE_TOLERANCE:
                continue

            row2_sorted = sorted(row2_all, key=lambda b: b[0])

            # Try to find 2 boxes matching row 2 width ratios (skip non-matching)
            row2 = _find_matching_boxes(
                row_boxes=row2_sorted,
                expected_ratios=PROFILE_ROW2_WIDTH_RATIOS,
                ref_aspect=PROFILE_ROW2_REF_ASPECT,
                expected_gap=expected_x_gap,
                gap_tolerance=gap_tolerance,
            )
            if row2 is None:
                continue

            # Verify row 1 and row 2 X coords align (first boxes must match)
            row1_x = row1[0][0]
            row2_x = row2[0][0]
            if abs(row1_x - row2_x) > gap_tolerance:
                continue

            # Calculate regiment box from row2, box2 (communications)
            # Relative coords: x=-1h, y=2.6h
            # Size: same width, 2/3 height
            row2_box2 = row2[1]
            x2, y2, w2, h2 = row2_box2
            regiment_box: Box = (
                x2 - h2,
                y2 + int(h2 * 2.6),
                w2,
                int(h2 * 2 / 3),
            )

            # Return only needed boxes: Username, Icon, Level, Regiment
            return [row1[0], row1[1], row1[2], regiment_box]

    return None


def _group_by_y(
    boxes: list[Box],
    ref_height: int,
) -> dict[int, list[Box]]:
    """Group boxes by Y position (same row).

    Args:
        boxes: List of boxes (x, y, w, h).
        ref_height: Reference height for tolerance calculation.

    Returns:
        Dictionary mapping reference Y to list of boxes in that row.
    """
    rows: dict[int, list[Box]] = {}

    for box in boxes:
        y = box[1]
        matched = False
        for ref_y in rows:
            if abs(y - ref_y) / ref_height <= PROFILE_TOLERANCE:
                rows[ref_y].append(box)
                matched = True
                break
        if not matched:
            rows[y] = [box]

    return rows
