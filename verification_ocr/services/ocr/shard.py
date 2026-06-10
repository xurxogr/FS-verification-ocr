"""Shard region detection for OCR service."""

from cv2.typing import MatLike

from verification_ocr.services.ocr.profile import PROFILE_REF_HEIGHT

# Box coordinates: (x, y, width, height)
type Box = tuple[int, int, int, int]

# The shard text block sits at a fixed location near the bottom-left corner.
# Cropping the exact block (region name / day-time / shard) instead of the whole
# bottom-left area keeps surrounding map terrain out of the crop, which would
# otherwise threshold into speckle and corrupt OCR line segmentation.
#
# Reference measurements taken at 1920x1080, where a profile box is
# PROFILE_REF_HEIGHT (35) px tall:
#   top-left corner: 28px right of the left edge, 112px up from the bottom edge
#   size: 180x48
# Expressed relative to profile_height so the crop scales with resolution.
SHARD_OFFSET_X_RATIO = 28 / PROFILE_REF_HEIGHT
SHARD_OFFSET_BOTTOM_RATIO = 112 / PROFILE_REF_HEIGHT
SHARD_CROP_WIDTH_RATIO = 180 / PROFILE_REF_HEIGHT
SHARD_CROP_HEIGHT_RATIO = 48 / PROFILE_REF_HEIGHT


def detect_shard_region(image: MatLike, profile_height: int) -> Box | None:
    """Detect the shard region near the bottom-left corner.

    The shard text block is at a fixed offset from the bottom-left corner.
    Its position and size are calculated from profile_height to handle
    different resolutions (1080p, 4K, etc.).

    Args:
        image (MatLike): BGR image to analyze.
        profile_height (int): Height of a profile box (from profile detection).

    Returns:
        Box | None: Box (x, y, w, h) for the shard region, or None if the
            region would fall outside the image bounds.
    """
    h, w = image.shape[:2]

    x = int(profile_height * SHARD_OFFSET_X_RATIO)
    y = h - int(profile_height * SHARD_OFFSET_BOTTOM_RATIO)
    crop_width = int(profile_height * SHARD_CROP_WIDTH_RATIO)
    crop_height = int(profile_height * SHARD_CROP_HEIGHT_RATIO)

    # Validate the region fits inside the image
    if y < 0 or x + crop_width > w or y + crop_height > h:
        return None

    return (x, y, crop_width, crop_height)
