"""Shard region detection for OCR service."""

from cv2.typing import MatLike

# Box coordinates: (x, y, width, height)
type Box = tuple[int, int, int, int]

# Crop ratios relative to profile height
# The shard box is always in the bottom-left corner
SHARD_CROP_WIDTH_RATIO = 10.7
SHARD_CROP_HEIGHT_RATIO = 3.5


def detect_shard_region(image: MatLike, profile_height: int) -> Box | None:
    """Detect the shard region in the bottom-left corner.

    The shard box is always located in the bottom-left corner of the image.
    Region dimensions are calculated based on profile_height to handle
    different resolutions (1080p, 4K, etc.).

    Args:
        image: BGR image to analyze.
        profile_height: Height of a profile box (from profile detection).

    Returns:
        Box (x, y, w, h) for the shard region, or None if image is too small.
    """
    h, w = image.shape[:2]

    crop_width = int(profile_height * SHARD_CROP_WIDTH_RATIO)
    crop_height = int(profile_height * SHARD_CROP_HEIGHT_RATIO)

    # Validate image is large enough
    if h < crop_height or w < crop_width:
        return None

    # Return box coordinates for the bottom-left region
    y_start = h - crop_height

    return (0, y_start, crop_width, crop_height)
