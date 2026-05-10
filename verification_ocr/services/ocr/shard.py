"""Shard detection for OCR service."""

from collections.abc import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

# Box coordinates: (x, y, width, height)
type Box = tuple[int, int, int, int]

# Reference dimensions at 1080p
PROFILE_REF_HEIGHT = 35
SHARD_REF_WIDTH = 340
SHARD_REF_HEIGHT = 92

# Derived ratios (relative to profile height)
SHARD_WIDTH_RATIO = SHARD_REF_WIDTH / PROFILE_REF_HEIGHT
SHARD_HEIGHT_RATIO = SHARD_REF_HEIGHT / PROFILE_REF_HEIGHT
SHARD_ASPECT_RATIO = SHARD_REF_WIDTH / SHARD_REF_HEIGHT

# Tolerance for size matching (5%)
SHARD_SIZE_TOLERANCE = 0.05

# BGR color ranges for shard box detection (brownish/tan tint)
# Format: (B, G, R) - OpenCV uses BGR order
# Each theme has a specific color range
SHARD_COLOR_RANGES = [
    # Dark theme: BGR varies across box (75-120, 100-150, 110-165)
    ((75, 100, 110), (120, 150, 165)),
    # Normal theme: BGR (174-180, 198-201, 205-208)
    ((170, 195, 200), (185, 205, 215)),
    # Light theme: BGR (190-195, 210-212, 216-218)
    ((185, 205, 210), (200, 220, 225)),
]


def detect_shard_box(image: MatLike, profile_height: int) -> Box | None:
    """Detect the shard box needed for verification.

    Uses the profile height to calculate expected shard box dimensions,
    then finds dark contours matching those dimensions.

    Args:
        image: BGR image to analyze.
        profile_height: Height of a profile box (from profile detection).

    Returns:
        Box (x, y, w, h) for the shard, or None if not found.
    """
    # Calculate expected shard dimensions based on profile height
    expected_width = profile_height * SHARD_WIDTH_RATIO
    expected_height = profile_height * SHARD_HEIGHT_RATIO

    # Try each color range and combine masks
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for bgr_min, bgr_max in SHARD_COLOR_RANGES:
        mask = cv2.inRange(
            src=image,
            lowerb=np.array(bgr_min),
            upperb=np.array(bgr_max),
        )
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    contours, _ = cv2.findContours(
        image=combined_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # Filter contours by expected dimensions
    candidates = _filter_by_size(
        contours=contours,
        expected_width=expected_width,
        expected_height=expected_height,
    )

    if not candidates:
        return None

    # Return the best match (closest to expected aspect ratio)
    return _select_best_match(candidates)


def _filter_by_size(
    contours: Sequence[MatLike],
    expected_width: float,
    expected_height: float,
) -> list[Box]:
    """Filter contours to those matching expected shard dimensions.

    Args:
        contours: Contours from cv2.findContours.
        expected_width: Expected box width based on profile scale.
        expected_height: Expected box height based on profile scale.

    Returns:
        List of boxes (x, y, w, h) matching expected dimensions.
    """
    boxes: list[Box] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check width within tolerance
        width_diff = abs(w - expected_width) / expected_width
        if width_diff > SHARD_SIZE_TOLERANCE:
            continue

        # Check height within tolerance
        height_diff = abs(h - expected_height) / expected_height
        if height_diff > SHARD_SIZE_TOLERANCE:
            continue

        boxes.append((x, y, w, h))

    return boxes


def _select_best_match(candidates: list[Box]) -> Box:
    """Select the best shard box candidate.

    Args:
        candidates: List of candidate boxes.

    Returns:
        The box with aspect ratio closest to expected.
    """
    best_box = candidates[0]
    best_diff = float("inf")

    for box in candidates:
        _, _, w, h = box
        aspect = w / h
        diff = abs(aspect - SHARD_ASPECT_RATIO)
        if diff < best_diff:
            best_diff = diff
            best_box = box

    return best_box
