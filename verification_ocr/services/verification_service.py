"""Verification service - OCR processing of image pairs."""

import logging
import os
import re
import time

import cv2
import numpy as np
import pytesseract
from cv2.typing import MatLike

from verification_ocr.core.settings import AppSettings
from verification_ocr.core.utils import calculate_war_time
from verification_ocr.enums import Faction
from verification_ocr.models import ImageRegions, Region, Verification
from verification_ocr.services.war_service import get_war_service

logger = logging.getLogger(__name__)

# Threshold for binary image conversion (same as foxhole_stockpiles)
TESSERACT_BINARY_THRESHOLD = 127

# Profile box dimensions at 2160p (4K reference resolution)
PROFILE_BOX_WIDTH_4K = 1096
PROFILE_BOX_HEIGHT_4K = 180
PROFILE_BOX_ASPECT_RATIO = PROFILE_BOX_WIDTH_4K / PROFILE_BOX_HEIGHT_4K  # ~6.09

# Grey box detection tolerances
GREY_BOX_Y_TOLERANCE = 3  # Pixels tolerance for grouping boxes by y position
GREY_VALUE_MIN = 20
GREY_VALUE_MAX = 100

# Grey box size constraints (for filtering candidates)
GREY_BOX_MIN_HEIGHT = 15
GREY_BOX_MAX_HEIGHT = 100
GREY_BOX_MIN_WIDTH = 30
GREY_BOX_MAX_WIDTH = 500

# Profile box detection thresholds
BLACK_THRESHOLD_VALUES = [15, 20, 25, 30]
PROFILE_BOX_MIN_WIDTH = 100
PROFILE_BOX_MIN_HEIGHT = 20
PROFILE_BOX_ASPECT_RATIO_TOLERANCE = 0.5
PROFILE_BOX_MAX_WIDTH_RATIO = 0.50  # Max ratio of box width to image width

# Grey box ratios within profile box (positions relative to box dimensions)
# Row 1 (username, icon, level, rank)
GREY_ROW1_Y_START = 0.10
GREY_ROW1_HEIGHT = 0.367  # 0.467 - 0.10
USERNAME_X_START = 0.022
USERNAME_WIDTH = 0.305  # 0.327 - 0.022
ICON_X_START = 0.389
ICON_WIDTH = 0.104  # 0.493 - 0.389
LEVEL_X_START = 0.506
LEVEL_WIDTH = 0.220  # 0.726 - 0.506
RANK_X_START = 0.739
RANK_WIDTH = 0.220  # 0.959 - 0.739

# Regiment region ratios (below profile box)
REGIMENT_X_START = 0.334
REGIMENT_X_END = 1.082
REGIMENT_Y_OFFSET = 0.539  # Below box bottom
REGIMENT_HEIGHT = 0.283

# Grey region to profile box expansion factors
GREY_TO_BOX_WIDTH_FACTOR = 1.05
GREY_TO_BOX_HEIGHT_FACTOR = 1.15
GREY_TO_BOX_X_OFFSET = 0.02
GREY_TO_BOX_Y_OFFSET = 0.05

# Template matching threshold
FACTION_MATCH_THRESHOLD = 0.7

# Minimum candidates for grey box pattern detection
MIN_GREY_BOX_CANDIDATES = 6
PROFILE_ROW1_BOX_COUNT = 4
PROFILE_ROW2_BOX_COUNT = 2

# Shard region positioning
SHARD_Y_OFFSET_MULTIPLIER = 3
SHARD_WIDTH_MULTIPLIER = 3.5

# Scale threshold for enabling OCR upscaling (below this, text is too small)
SCALE_UPSCALE_THRESHOLD = 0.9

# Row distance tolerance factor for grey box pattern detection
ROW_DISTANCE_TOLERANCE_FACTOR = 1.15


def extract_day_and_hour(text: str) -> str:
    """
    Extract days and hours from a formatted string.

    Args:
        text (str): Input text containing numbers and commas.

    Returns:
        str: Formatted string with days and hours, e.g. "1234, 15:30".
    """
    # Find all digit/comma groups and join
    result = "".join(re.findall(pattern=r"[\d,]+", string=text))
    # Remove first comma if exactly two commas
    if result.count(",") == 2:
        result = result.replace(",", "", 1)
    # Try to split into left/right by first comma
    parts = result.split(",", 1)
    if len(parts) == 2:
        left, right = parts
        digits = re.sub(pattern=r"\D", repl="", string=right)
        if len(digits) == 4:
            return f"{left}, {digits[:2]}:{digits[2:]}"
    return result


def parse_ingame_time(ingame_time: str) -> tuple[int, int, int] | None:
    """
    Parse in-game time string to day, hour, minute.

    Args:
        ingame_time (str): Time string like "267, 21:45".

    Returns:
        tuple[int, int, int] | None: (day, hour, minute) or None if parsing fails.
    """
    try:
        # Format: "267, 21:45"
        parts = ingame_time.split(", ")
        if len(parts) != 2:
            return None

        day = int(parts[0].strip())
        # Validate day is within reasonable bounds (1-9999)
        if day < 1 or day > 9999:
            return None

        time_parts = parts[1].split(":")
        if len(time_parts) != 2:
            return None

        hour = int(time_parts[0])
        minute = int(time_parts[1])

        # Validate hour (0-23) and minute (0-59) ranges
        if not (0 <= hour <= 23):
            return None
        if not (0 <= minute <= 59):
            return None

        return day, hour, minute
    except (ValueError, IndexError):
        return None


def calculate_ingame_time_diff(
    extracted_day: int,
    extracted_hour: int,
    current_day: int,
    current_hour: int,
) -> int:
    """
    Calculate the absolute difference in in-game hours between two times.

    Args:
        extracted_day (int): Day from screenshot.
        extracted_hour (int): Hour from screenshot.
        current_day (int): Current calculated day.
        current_hour (int): Current calculated hour.

    Returns:
        int: Absolute difference in in-game hours.
    """
    # Convert both times to total in-game hours
    extracted_total = (extracted_day * 24) + extracted_hour
    current_total = (current_day * 24) + current_hour

    return abs(extracted_total - current_total)


def get_current_ingame_time() -> tuple[int, int, int] | None:
    """
    Get the current in-game time from war state.

    Returns:
        tuple[int, int, int] | None: (day, hour, minute) or None if not configured.
    """
    war_service = get_war_service()

    if war_service.state.start_time is None:
        return None

    return calculate_war_time(start_time=war_service.state.start_time)


class VerificationService:
    """Service for processing and comparing two images using OCR."""

    def __init__(self, settings: AppSettings) -> None:
        """
        Initialize the verification service.

        Args:
            settings (AppSettings): Application settings instance.
        """
        self.settings = settings
        if settings.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.ocr.tesseract_cmd

        # Load faction icons if paths are provided
        self.colonial_icon: cv2.typing.MatLike | None = None
        if settings.ocr.colonial_icon_path:
            self.colonial_icon = cv2.imread(
                filename=settings.ocr.colonial_icon_path,
                flags=cv2.IMREAD_COLOR,
            )
            if self.colonial_icon is None:
                logger.warning(
                    f"Failed to load colonial icon from {settings.ocr.colonial_icon_path}"
                )

        self.warden_icon: cv2.typing.MatLike | None = None
        if settings.ocr.warden_icon_path:
            self.warden_icon = cv2.imread(
                filename=settings.ocr.warden_icon_path,
                flags=cv2.IMREAD_COLOR,
            )
            if self.warden_icon is None:
                logger.warning(f"Failed to load warden icon from {settings.ocr.warden_icon_path}")

        # Create debug output directory if debug mode is enabled
        if settings.ocr.debug_mode:
            os.makedirs(settings.ocr.debug_output_dir, exist_ok=True)

        # Cache CLAHE object for image enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=settings.ocr.clahe_clip_limit,
            tileGridSize=(
                settings.ocr.clahe_grid_size,
                settings.ocr.clahe_grid_size,
            ),
        )

    def _extract_text_from_image(
        self,
        image: cv2.typing.MatLike,
        scale: bool = False,
        use_invert: bool = True,
    ) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image (cv2.typing.MatLike): Image to extract text from.
            scale (bool): Whether to scale the image before extracting text.
            use_invert (bool): Whether to invert the image colors.

        Returns:
            str: Extracted text.
        """
        if image is None or image.size == 0:
            return ""

        if scale:
            scale_factor = self.settings.ocr.scale_factor
            resized = cv2.resize(
                src=image,
                dsize=None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            resized = image

        # Convert to grayscale
        gray = cv2.cvtColor(src=resized, code=cv2.COLOR_BGR2GRAY)

        # Enhance contrast using cached CLAHE
        enhanced = self.clahe.apply(gray)

        if use_invert:
            processed = cv2.bitwise_not(enhanced)
        else:
            processed = enhanced

        config = "--psm 7"
        text: str = pytesseract.image_to_string(
            image=processed,
            config=config,
            lang=self.settings.ocr.language,
        )
        return text.strip()

    def _prepare_image_for_shard_detection(
        self,
        image: cv2.typing.MatLike,
        scale_factor: float,
    ) -> cv2.typing.MatLike:
        """
        Prepare image for shard/time text detection.

        Uses Otsu's thresholding (same as foxhole-stockpiles) to automatically
        determine optimal threshold based on image histogram.

        Args:
            image (cv2.typing.MatLike): Image region to preprocess.
            scale_factor (float): Resolution scale factor.

        Returns:
            cv2.typing.MatLike: Processed image ready for OCR.
        """
        if image is None or image.size == 0:
            return image

        # Convert to grayscale
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        # Upscale based on scale factor (at 4K stays same, at 1080p doubles)
        upscale_factor = 2 / scale_factor
        upscaled = cv2.resize(
            src=gray,
            dsize=None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        # Use Otsu's thresholding (THRESH_BINARY for white text on dark background)
        # This automatically calculates the optimal threshold from the histogram
        threshold_mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, binary = cv2.threshold(
            src=upscaled,
            thresh=0,
            maxval=255,
            type=threshold_mode,
        )

        # Dilate to connect text components
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        binary = cv2.dilate(src=binary, kernel=kernel, iterations=1)

        # Convert back to RGB for tesseract
        result = cv2.cvtColor(src=binary, code=cv2.COLOR_GRAY2RGB)

        return np.asarray(a=result, dtype=np.uint8)

    def _match_template(
        self,
        image: cv2.typing.MatLike,
        template: cv2.typing.MatLike,
    ) -> float:
        """
        Match a template against an image region.

        Args:
            image (cv2.typing.MatLike): Image to search in.
            template (cv2.typing.MatLike): Template to search for.

        Returns:
            float: Match confidence value (0.0 to 1.0).
        """
        # Calculate the scale to match the height of the image
        scale = image.shape[0] / template.shape[0]

        # Resize the template according to the scale
        resized_template = cv2.resize(
            src=template,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        # Perform template matching
        result = cv2.matchTemplate(
            image=image,
            templ=resized_template,
            method=cv2.TM_CCOEFF_NORMED,
        )

        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val

    def _find_profile_box(
        self,
        image: cv2.typing.MatLike,
    ) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]] | None:
        """
        Detect the black profile box in the image.

        Uses two detection methods:
        1. Direct black box detection (for full screenshots)
        2. Grey sub-box pattern detection (for cropped images)

        After detection, validates by checking for a faction icon inside.

        Args:
            image: Image to search in.

        Returns:
            Tuple of (box, grey_boxes) or None if not found.
            box is (x, y, width, height) of the profile box.
            grey_boxes is list of 4 tuples [username, icon, level, rank].
        """
        # Try direct black box detection first
        box = self._find_profile_box_by_black(image)
        grey_boxes = None

        # Fallback: detect grey sub-boxes and calculate profile box from them
        if box is None:
            result = self._find_grey_boxes_pattern(image)
            if result is not None:
                box, grey_boxes = result

        # Validate by checking for faction icon inside
        if box is None:
            return None

        # Calculate grey boxes from black box if not directly detected
        if grey_boxes is None:
            grey_boxes = self._calculate_grey_boxes_from_black_box(box)

        # grey_boxes[1] is the icon grey box
        icon_box = grey_boxes[1]
        icon_region = Region(
            x1=icon_box[0],
            y1=icon_box[1],
            x2=icon_box[0] + icon_box[2],
            y2=icon_box[1] + icon_box[3],
        )

        faction = self._detect_faction_with_scaled_template(
            image=image,
            icon_region=icon_region,
            threshold=FACTION_MATCH_THRESHOLD,
        )
        if faction is None:
            return None

        return (box, grey_boxes)

    def _find_profile_box_by_black(
        self,
        image: cv2.typing.MatLike,
    ) -> tuple[int, int, int, int] | None:
        """
        Detect profile box by finding black rectangle with correct aspect ratio.

        Args:
            image: Image to search in.

        Returns:
            Tuple (x, y, width, height) or None if not found.
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        best_match = None
        best_score = float("inf")

        for thresh in BLACK_THRESHOLD_VALUES:
            _, black_mask = cv2.threshold(
                src=gray,
                thresh=thresh,
                maxval=255,
                type=cv2.THRESH_BINARY_INV,
            )
            contours, _ = cv2.findContours(
                image=black_mask,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < PROFILE_BOX_MIN_WIDTH or h < PROFILE_BOX_MIN_HEIGHT or h == 0:
                    continue

                ratio = w / h
                ratio_diff = abs(ratio - PROFILE_BOX_ASPECT_RATIO)

                if ratio_diff >= PROFILE_BOX_ASPECT_RATIO_TOLERANCE:
                    continue

                # Reject boxes at origin or too large (merging with background)
                if x == 0 and y == 0:
                    continue
                if w / img_w > PROFILE_BOX_MAX_WIDTH_RATIO:
                    continue

                if ratio_diff < best_score:
                    best_score = ratio_diff
                    best_match = (x, y, w, h)

        return best_match

    def _find_grey_boxes_pattern(
        self,
        image: cv2.typing.MatLike,
    ) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]] | None:
        """
        Find the 4+2 grey box pattern and return both the profile box and grey boxes.

        Looks for 4 grey boxes in row 1 followed by 2 grey boxes in row 2.

        Args:
            image: Image to search in.

        Returns:
            Tuple of (profile_box, grey_boxes) or None if not found.
            grey_boxes contains [username, icon, level, rank] sorted by x position.
        """
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        # Find grey regions (profile sub-boxes are grey)
        grey_mask = cv2.inRange(
            src=gray,
            lowerb=np.array([GREY_VALUE_MIN]),
            upperb=np.array([GREY_VALUE_MAX]),
        )
        contours, _ = cv2.findContours(
            image=grey_mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter for rectangular regions that could be profile sub-boxes
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (
                GREY_BOX_MIN_HEIGHT < h < GREY_BOX_MAX_HEIGHT
                and GREY_BOX_MIN_WIDTH < w < GREY_BOX_MAX_WIDTH
            ):
                candidates.append((x, y, w, h))

        if len(candidates) < MIN_GREY_BOX_CANDIDATES:
            return None

        # Group by similar y position
        rows: dict[int, list[tuple[int, int, int, int]]] = {}
        for box in candidates:
            x, y, w, h = box
            row_key = None
            for ry in rows:
                if abs(ry - y) < GREY_BOX_Y_TOLERANCE:
                    row_key = ry
                    break
            if row_key is None:
                row_key = y
                rows[row_key] = []
            rows[row_key].append(box)

        # Look for 4+2 pattern: row with 4 boxes followed by row with 2 boxes
        sorted_row_ys = sorted(rows.keys())

        for i, row_y in enumerate(sorted_row_ys):
            if len(rows[row_y]) != PROFILE_ROW1_BOX_COUNT:
                continue

            row1_boxes = rows[row_y]
            row2_y = self._find_row2_below(
                row1_y=row_y,
                row1_boxes=row1_boxes,
                sorted_row_ys=sorted_row_ys[i + 1 :],
                rows=rows,
            )

            if row2_y is None:
                continue

            row2_boxes = rows[row2_y]
            return self._calculate_profile_box_from_rows(row1_boxes, row2_boxes)

        return None

    def _find_row2_below(
        self,
        row1_y: int,
        row1_boxes: list[tuple[int, int, int, int]],
        sorted_row_ys: list[int],
        rows: dict[int, list[tuple[int, int, int, int]]],
    ) -> int | None:
        """
        Find the second row (2 boxes) below the first row (4 boxes).

        Args:
            row1_y: Y position of row 1.
            row1_boxes: Boxes in row 1.
            sorted_row_ys: Remaining row Y positions to check.
            rows: All rows grouped by Y position.

        Returns:
            Y position of row 2 or None if not found.
        """
        avg_box_height = sum(b[3] for b in row1_boxes) // len(row1_boxes)
        max_row_distance = int(avg_box_height * ROW_DISTANCE_TOLERANCE_FACTOR)

        for next_row_y in sorted_row_ys:
            if next_row_y - row1_y > max_row_distance:
                break
            if len(rows[next_row_y]) == PROFILE_ROW2_BOX_COUNT:
                return next_row_y

        return None

    def _calculate_profile_box_from_rows(
        self,
        row1_boxes: list[tuple[int, int, int, int]],
        row2_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
        """
        Calculate profile box bounds from the detected grey box rows.

        Args:
            row1_boxes: 4 boxes from row 1 (username, icon, level, rank).
            row2_boxes: 2 boxes from row 2 (commends, communications).

        Returns:
            Tuple of (profile_box, row1_sorted) where row1_sorted is sorted by x.
        """
        all_boxes = row1_boxes + row2_boxes
        row1_sorted = sorted(row1_boxes, key=lambda b: b[0])

        x_min = min(b[0] for b in all_boxes)
        x_max = max(b[0] + b[2] for b in all_boxes)
        y_min = min(b[1] for b in all_boxes)
        y_max = max(b[1] + b[3] for b in all_boxes)

        grey_width = x_max - x_min
        grey_height = y_max - y_min

        box_w = int(grey_width * GREY_TO_BOX_WIDTH_FACTOR)
        box_h = int(grey_height * GREY_TO_BOX_HEIGHT_FACTOR)
        box_x = x_min - int(box_w * GREY_TO_BOX_X_OFFSET)
        box_y = y_min - int(box_h * GREY_TO_BOX_Y_OFFSET)

        profile_box = (max(0, box_x), max(0, box_y), box_w, box_h)
        return (profile_box, row1_sorted)

    def _calculate_grey_boxes_from_black_box(
        self,
        box: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """
        Calculate grey box positions from the black profile box.

        The grey boxes are at fixed ratios within the black box.
        Returns [username, icon, level, rank] boxes.

        Args:
            box: (x, y, width, height) of the black profile box.

        Returns:
            List of 4 grey box tuples (x, y, w, h) for [username, icon, level, rank].
        """
        box_x, box_y, box_width, box_height = box

        row1_y = box_y + int(box_height * GREY_ROW1_Y_START)
        row1_height = int(box_height * GREY_ROW1_HEIGHT)

        username = (
            box_x + int(box_width * USERNAME_X_START),
            row1_y,
            int(box_width * USERNAME_WIDTH),
            row1_height,
        )

        icon = (
            box_x + int(box_width * ICON_X_START),
            row1_y,
            int(box_width * ICON_WIDTH),
            row1_height,
        )

        level = (
            box_x + int(box_width * LEVEL_X_START),
            row1_y,
            int(box_width * LEVEL_WIDTH),
            row1_height,
        )

        rank = (
            box_x + int(box_width * RANK_X_START),
            row1_y,
            int(box_width * RANK_WIDTH),
            row1_height,
        )

        return [username, icon, level, rank]

    def _calculate_regions_from_grey_boxes(
        self,
        image: cv2.typing.MatLike,
        box: tuple[int, int, int, int],
        grey_boxes: list[tuple[int, int, int, int]],
    ) -> ImageRegions:
        """
        Calculate region coordinates from grey box positions.

        Args:
            image: The image.
            box: (x, y, width, height) of the profile box.
            grey_boxes: List of 4 grey boxes [username, icon, level, rank] sorted by x.

        Returns:
            ImageRegions with coordinates from grey box positions.
        """
        img_height, img_width = image.shape[:2]
        box_x, box_y, box_width, box_height = box

        # Scale factor: box_height / reference height (180 is box height at 2160p)
        scale_factor = box_height / PROFILE_BOX_HEIGHT_4K

        # Use grey box positions for username, icon, level
        username_box = grey_boxes[0]
        icon_box = grey_boxes[1]
        level_box = grey_boxes[2]

        # Regiment is below the profile box - calculate from box dimensions
        regiment_x1 = box_x + int(box_width * REGIMENT_X_START)
        regiment_x2 = min(img_width, box_x + int(box_width * REGIMENT_X_END))
        regiment_y1 = box_y + box_height + int(box_height * REGIMENT_Y_OFFSET)
        regiment_y2 = regiment_y1 + int(box_height * REGIMENT_HEIGHT)

        # Shard region (for the other image, bottom-left)
        shard_box_width = int(self.settings.ocr.base_box_width * scale_factor)
        shard_box_height = int(self.settings.ocr.base_box_height * scale_factor)
        shard_x = shard_box_height
        shard_y = img_height - int(shard_box_height * SHARD_Y_OFFSET_MULTIPLIER)
        shard_width = int(shard_box_width * SHARD_WIDTH_MULTIPLIER)

        return ImageRegions(
            username=Region(
                x1=username_box[0],
                y1=username_box[1],
                x2=username_box[0] + username_box[2],
                y2=username_box[1] + username_box[3],
            ),
            icon=Region(
                x1=icon_box[0],
                y1=icon_box[1],
                x2=icon_box[0] + icon_box[2],
                y2=icon_box[1] + icon_box[3],
            ),
            level=Region(
                x1=level_box[0],
                y1=level_box[1],
                x2=level_box[0] + level_box[2],
                y2=level_box[1] + level_box[3],
            ),
            regiment=Region(x1=regiment_x1, y1=regiment_y1, x2=regiment_x2, y2=regiment_y2),
            shard=Region(
                x1=shard_x,
                y1=shard_y,
                x2=shard_x + shard_width,
                y2=shard_y + shard_box_height,
            ),
            scale_factor=scale_factor,
        )

    def _detect_faction_with_scaled_template(
        self,
        image: cv2.typing.MatLike,
        icon_region: Region,
        threshold: float,
    ) -> Faction | None:
        """
        Detect faction by matching scaled master icons against the icon region.

        Args:
            image: The full image.
            icon_region: The region where the icon should be.
            threshold: Minimum match confidence to accept.

        Returns:
            Detected Faction or None if no match.
        """
        # Extract icon region from image
        icon_img = image[icon_region.y1 : icon_region.y2, icon_region.x1 : icon_region.x2]
        if icon_img.size == 0:
            return None

        # Icon is square, sized to fit the region (use smaller dimension)
        region_h = icon_region.y2 - icon_region.y1
        region_w = icon_region.x2 - icon_region.x1
        icon_size = min(region_h, region_w)

        best_faction = None
        best_score = threshold

        for template, faction in [
            (self.colonial_icon, Faction.COLONIAL),
            (self.warden_icon, Faction.WARDEN),
        ]:
            if template is None:
                continue

            # Scale master icon to fit region
            scaled_template = cv2.resize(
                src=template,
                dsize=(icon_size, icon_size),
                interpolation=cv2.INTER_CUBIC,
            )

            # Template must fit in the icon region
            if (
                scaled_template.shape[0] > icon_img.shape[0]
                or scaled_template.shape[1] > icon_img.shape[1]
            ):
                continue

            result = cv2.matchTemplate(
                image=icon_img,
                templ=scaled_template,
                method=cv2.TM_CCOEFF_NORMED,
            )

            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_faction = faction

        return best_faction

    def _find_shard_dynamic(
        self,
        image: cv2.typing.MatLike,
    ) -> tuple[str | None, str | None]:
        """
        Find shard and time dynamically by searching for the time pattern.

        The shard name is always on the line immediately after the time.
        Time format: digits + comma + 4 digits (e.g., "Day 154, 2335 Hours").

        Args:
            image: Shard/map image to extract from.

        Returns:
            Tuple of (shard, ingame_time) or (None, None) if not found.
        """
        # OCR the entire image
        text = pytesseract.image_to_string(image, lang=self.settings.ocr.language)

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        found_time: str | None = None
        found_shard: str | None = None

        # Find the line with time pattern, shard is the next line
        for i, line in enumerate(lines):
            # Look for pattern: digits + comma + 4 digits (e.g., "154, 2335" or "Day 154, 2335")
            if re.search(r"\d{1,4}\s*,\s*\d{4}", line):
                found_time = extract_day_and_hour(line)
                if found_time and "," in found_time and ":" in found_time:
                    # Shard is on the next line
                    if i + 1 < len(lines):
                        found_shard = lines[i + 1].upper().strip()
                    break

        return found_shard, found_time

    def _detect_faction(
        self,
        image: cv2.typing.MatLike,
        threshold: float,
    ) -> Faction | None:
        """
        Detect faction from the icon region using template matching.

        Matches against both colonial and wardens icons and returns the faction
        with the highest confidence above threshold.

        Args:
            image (cv2.typing.MatLike): Icon region image.
            threshold: Minimum match confidence to accept.

        Returns:
            Faction | None: Detected faction, or None if no match above threshold.
        """
        colonial_score = 0.0
        wardens_score = 0.0

        if self.colonial_icon is not None:
            colonial_score = self._match_template(image=image, template=self.colonial_icon)

        if self.warden_icon is not None:
            wardens_score = self._match_template(image=image, template=self.warden_icon)

        # Return the faction with the highest score above threshold
        if colonial_score >= threshold and colonial_score >= wardens_score:
            return Faction.COLONIAL
        if wardens_score >= threshold and wardens_score > colonial_score:
            return Faction.WARDEN

        # No faction detected above threshold
        return None

    def _save_debug_image(
        self,
        image: cv2.typing.MatLike,
        regions: ImageRegions,
        filename: str,
    ) -> None:
        """
        Save debug image with detected regions drawn.

        Args:
            image (cv2.typing.MatLike): Original image.
            regions (ImageRegions): Region coordinates.
            filename (str): Output filename (must be alphanumeric with underscores/hyphens).
        """
        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(filename)
        # Only allow alphanumeric, underscores, hyphens, and dots
        if not re.match(r"^[\w\-\.]+$", safe_filename):
            logger.warning(f"Invalid debug filename rejected: {filename}")
            return

        # Verify output path is within debug directory
        debug_dir = os.path.realpath(self.settings.ocr.debug_output_dir)
        output_path = os.path.realpath(os.path.join(debug_dir, safe_filename))
        if not output_path.startswith(debug_dir + os.sep):
            logger.warning(f"Path traversal attempt rejected: {filename}")
            return

        debug_img = image.copy()
        scale = regions.scale_factor
        thickness = max(1, int(2 * scale))

        # Draw username region (green)
        r = regions.username
        cv2.rectangle(
            img=debug_img,
            pt1=(r.x1, r.y1),
            pt2=(r.x2, r.y2),
            color=(0, 255, 0),
            thickness=thickness,
        )
        cv2.putText(
            img=debug_img,
            text="USERNAME",
            org=(r.x1, r.y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale * 0.5,
            color=(0, 255, 0),
            thickness=thickness,
        )

        # Draw icon region (yellow)
        r = regions.icon
        cv2.rectangle(
            img=debug_img,
            pt1=(r.x1, r.y1),
            pt2=(r.x2, r.y2),
            color=(0, 255, 255),
            thickness=thickness,
        )
        cv2.putText(
            img=debug_img,
            text="ICON",
            org=(r.x1, r.y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale * 0.5,
            color=(0, 255, 255),
            thickness=thickness,
        )

        # Draw level region (orange)
        r = regions.level
        cv2.rectangle(
            img=debug_img,
            pt1=(r.x1, r.y1),
            pt2=(r.x2, r.y2),
            color=(0, 165, 255),
            thickness=thickness,
        )
        cv2.putText(
            img=debug_img,
            text="LEVEL",
            org=(r.x1, r.y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale * 0.5,
            color=(0, 165, 255),
            thickness=thickness,
        )

        # Draw regiment region (magenta)
        r = regions.regiment
        cv2.rectangle(
            img=debug_img,
            pt1=(r.x1, r.y1),
            pt2=(r.x2, r.y2),
            color=(255, 0, 255),
            thickness=thickness,
        )
        cv2.putText(
            img=debug_img,
            text="REGIMENT",
            org=(r.x1, r.y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale * 0.5,
            color=(255, 0, 255),
            thickness=thickness,
        )

        # Draw shard region (cyan)
        r = regions.shard
        cv2.rectangle(
            img=debug_img,
            pt1=(r.x1, r.y1),
            pt2=(r.x2, r.y2),
            color=(255, 255, 0),
            thickness=thickness,
        )
        cv2.putText(
            img=debug_img,
            text="SHARD/TIME",
            org=(r.x1, r.y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale * 0.5,
            color=(255, 255, 0),
            thickness=thickness,
        )

        cv2.imwrite(filename=output_path, img=debug_img)
        logger.info(f"Debug image saved to {output_path}")

    def _find_user_info(
        self,
        image: cv2.typing.MatLike,
        regions: ImageRegions,
    ) -> Verification:
        """
        Extract user information from the image.

        Extracts name, level, regiment status, and faction.

        Args:
            image (cv2.typing.MatLike): Image to extract user information from.
            regions (ImageRegions): Pre-calculated region coordinates.

        Returns:
            Verification: User information.
        """
        data = Verification()

        # Enable scaling for lower resolution images (below 4K)
        # This improves OCR accuracy for smaller text
        use_scale = regions.scale_factor < SCALE_UPSCALE_THRESHOLD

        # Extract username from dedicated region
        r = regions.username
        username_image = image[r.y1 : r.y2, r.x1 : r.x2]
        username_text = self._extract_text_from_image(username_image, scale=use_scale)
        data.name = username_text.strip() if username_text.strip() else None

        # No name found - either image is too small or this is a map image
        if data.name is None:
            return data

        # Extract level from dedicated region (language-agnostic: look for : and digits)
        r = regions.level
        level_image = image[r.y1 : r.y2, r.x1 : r.x2]
        level_text = self._extract_text_from_image(level_image, scale=use_scale)
        # Look for colon and extract digits after it
        if ":" in level_text:
            level_part = level_text.split(":")[-1]
            digits = "".join(c for c in level_part if c.isdigit())
            if digits:
                data.level = int(digits)

        # Note: Faction is detected separately using scaled template matching
        # in _detect_faction_with_scaled_template() and set by the caller

        # Extract Regiment name from grey bar region
        r = regions.regiment
        regiment_image = image[r.y1 : r.y2, r.x1 : r.x2]
        regiment_text = self._extract_text_from_image(regiment_image, scale=use_scale)
        data.regiment = self._parse_regiment_name(regiment_text)

        return data

    def _parse_regiment_name(self, text: str) -> str | None:
        """
        Parse regiment name from OCR text.

        Format: [TAG#NUM] Regiment Name (something) - we extract [TAG#NUM] Regiment Name

        Args:
            text (str): Raw OCR text from regiment name region.

        Returns:
            str | None: Parsed regiment name or None if not found.
        """
        if not text:
            return None

        # Clean up the text
        text = text.strip()

        # Look for regiment tag format: [TAG#NUM] Name
        # Find the start of the tag
        bracket_start = text.find("[")
        if bracket_start == -1:
            return None

        # Extract from the bracket onwards
        text = text[bracket_start:]

        # Remove anything after ( as it's usually "(Players)" or similar UI text
        if "(" in text:
            text = text.split("(")[0].strip()

        # Remove anything after | as it could be "| Players" tab
        if "|" in text:
            text = text.split("|")[0].strip()

        # Remove newlines and extra spaces
        text = " ".join(text.split())

        # Strip spaces from the tag portion (between [ and ])
        # OCR often introduces spurious spaces in the tag like "[7-HP #8707]" -> "[7-HP#8707]"
        bracket_end = text.find("]")
        if bracket_end != -1:
            tag = text[1:bracket_end]  # Content between [ and ]
            tag_no_spaces = tag.replace(" ", "")
            text = "[" + tag_no_spaces + "]" + text[bracket_end + 1 :]

        return text if text else None

    def _get_shard_and_time(
        self,
        image: cv2.typing.MatLike,
        regions: ImageRegions,
    ) -> tuple[str | None, str | None]:
        """
        Extract shard and in-game time from the image.

        The shard region contains both timestamp (line 1) and shard name (line 2).

        Args:
            image (cv2.typing.MatLike): Image to extract shard from.
            regions (ImageRegions): Pre-calculated region coordinates.

        Returns:
            tuple[str | None, str | None]: (shard, ingame_time) tuple.
        """
        r = regions.shard
        shard_image = image[r.y1 : r.y2, r.x1 : r.x2]

        if shard_image.size == 0:
            return None, None

        # Preprocess using foxhole_stockpiles method
        scale_factor = regions.scale_factor
        processed_image = self._prepare_image_for_shard_detection(
            image=shard_image,
            scale_factor=scale_factor,
        )

        # Save debug image of processed shard region
        if self.settings.ocr.debug_mode:
            output_path = os.path.join(
                self.settings.ocr.debug_output_dir,
                "debug_shard_processed.png",
            )
            cv2.imwrite(filename=output_path, img=processed_image)

        # Extract text with PSM 6 for block of text (multiple lines)
        config = "--psm 6"
        text = pytesseract.image_to_string(
            image=processed_image,
            config=config,
            lang=self.settings.ocr.language,
        )
        text = text.strip()

        if not text:
            return None, None

        lines = text.split("\n")
        ingame_time = extract_day_and_hour(lines[0]) if lines else None
        shard = lines[1].strip() if len(lines) > 1 else None

        return shard, ingame_time

    def _decode_and_validate_images(
        self,
        image1_bytes: bytes,
        image2_bytes: bytes,
    ) -> tuple[MatLike, MatLike]:
        """
        Decode image bytes and validate minimum dimensions.

        Args:
            image1_bytes: First image bytes.
            image2_bytes: Second image bytes.

        Returns:
            Tuple of decoded images (img1, img2).

        Raises:
            ValueError: If images cannot be decoded or are too small.
        """
        images: list[MatLike | None] = []
        for img_bytes in [image1_bytes, image2_bytes]:
            nparr = np.frombuffer(buffer=img_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
            images.append(img)

        if images[0] is None or images[1] is None:
            raise ValueError("Failed to decode one or both images")

        img1: MatLike = images[0]
        img2: MatLike = images[1]

        for i, img in enumerate([img1, img2]):
            if img.shape[0] < 100 or img.shape[1] < 100:
                raise ValueError(f"Image {i + 1} is too small (minimum 100x100 pixels)")

        return img1, img2

    def _determine_profile_and_shard_images(
        self,
        img1: MatLike,
        img2: MatLike,
    ) -> tuple[MatLike, MatLike, tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
        """
        Determine which image is profile and which is shard based on profile box detection.

        Args:
            img1: First decoded image.
            img2: Second decoded image.

        Returns:
            Tuple of (profile_img, shard_img, profile_box, grey_boxes).

        Raises:
            ValueError: If both or neither images have profile boxes.
        """
        result1 = self._find_profile_box(img1)
        result2 = self._find_profile_box(img2)

        logger.debug(
            f"Profile box detection: img1={'found' if result1 else 'not found'}, "
            f"img2={'found' if result2 else 'not found'}"
        )

        if result1 is not None and result2 is None:
            profile_box, grey_boxes = result1
            return img1, img2, profile_box, grey_boxes
        elif result2 is not None and result1 is None:
            profile_box, grey_boxes = result2
            return img2, img1, profile_box, grey_boxes
        elif result1 is not None and result2 is not None:
            raise ValueError(
                "Both images appear to be profile images (both have profile box). "
                "Expected one profile image and one shard/map image."
            )
        else:
            raise ValueError(
                "Could not identify profile image - no profile box detected in either image. "
                "Expected one image with the player profile UI."
            )

    def _extract_profile_data(
        self,
        profile_img: MatLike,
        profile_box: tuple[int, int, int, int],
        grey_boxes: list[tuple[int, int, int, int]] | None,
    ) -> tuple[Verification, ImageRegions]:
        """
        Extract verification data from the profile image.

        Args:
            profile_img: The profile image.
            profile_box: The detected profile box coordinates.
            grey_boxes: List of grey boxes or None to calculate from black box.

        Returns:
            Tuple of (verification data, profile regions).

        Raises:
            ValueError: If no player name found.
        """
        if grey_boxes is None:
            grey_boxes = self._calculate_grey_boxes_from_black_box(box=profile_box)

        profile_regions = self._calculate_regions_from_grey_boxes(
            image=profile_img,
            box=profile_box,
            grey_boxes=grey_boxes,
        )

        faction = self._detect_faction_with_scaled_template(
            image=profile_img,
            icon_region=profile_regions.icon,
            threshold=FACTION_MATCH_THRESHOLD,
        )

        if self.settings.ocr.debug_mode:
            self._save_debug_image(
                image=profile_img,
                regions=profile_regions,
                filename="debug_profile_regions.png",
            )

        verification = self._find_user_info(image=profile_img, regions=profile_regions)
        verification.faction = faction

        if verification.name is None:
            raise ValueError("No player name found in the profile image")

        return verification, profile_regions

    def _extract_shard_data(
        self,
        shard_img: MatLike,
        verification: Verification,
    ) -> Verification:
        """
        Extract shard and war data and add to verification.

        Args:
            shard_img: The shard/map image.
            verification: Verification object to update.

        Returns:
            Updated verification with shard and war data.

        Raises:
            ValueError: If no shard information found.
        """
        shard, ingame_time = self._find_shard_dynamic(shard_img)

        if shard is None:
            raise ValueError("No shard information found in the map/shard image")

        verification.shard = shard
        verification.ingame_time = ingame_time

        war_service = get_war_service()
        verification.war_number = war_service.state.war_number

        current_time = get_current_ingame_time()
        if current_time is not None:
            current_day, current_hour, current_minute = current_time
            verification.current_ingame_time = (
                f"{current_day}, {current_hour:02d}:{current_minute:02d}"
            )

        return verification

    def verify(self, image1_bytes: bytes, image2_bytes: bytes) -> Verification:
        """
        Process two images and extract user verification information.

        Args:
            image1_bytes: First image bytes.
            image2_bytes: Second image bytes.

        Returns:
            Verification: Extracted verification data.

        Raises:
            ValueError: If images cannot be decoded or are invalid.
            RuntimeError: If image processing fails.
        """
        logger.info("Processing image pair for verification")
        start_time = time.perf_counter()

        try:
            decode_start = time.perf_counter()
            img1, img2 = self._decode_and_validate_images(
                image1_bytes=image1_bytes,
                image2_bytes=image2_bytes,
            )
            decode_time = time.perf_counter() - decode_start

            ocr_start = time.perf_counter()
            profile_img, shard_img, profile_box, grey_boxes = (
                self._determine_profile_and_shard_images(img1=img1, img2=img2)
            )

            verification, _ = self._extract_profile_data(
                profile_img=profile_img,
                profile_box=profile_box,
                grey_boxes=grey_boxes,
            )

            verification = self._extract_shard_data(
                shard_img=shard_img,
                verification=verification,
            )
            ocr_time = time.perf_counter() - ocr_start

            total_time = time.perf_counter() - start_time
            logger.info(
                f"Verification completed in {total_time:.3f}s "
                f"(decode: {decode_time:.3f}s, ocr: {ocr_time:.3f}s)"
            )

            return verification

        except cv2.error as e:
            logger.error(f"OpenCV error during verification: {e}")
            raise RuntimeError("Image processing error") from e
        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract OCR error: {e}")
            raise RuntimeError("OCR processing error") from e
        except ValueError:
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error during verification",
                extra={
                    "image1_size": len(image1_bytes),
                    "image2_size": len(image2_bytes),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeError("Internal processing error") from e
