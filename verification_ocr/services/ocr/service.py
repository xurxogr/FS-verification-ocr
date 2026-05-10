"""OCR service for image text extraction."""

import logging
import os
import re
from typing import TypedDict

import cv2
import numpy as np
import pytesseract
from cv2.typing import MatLike

from verification_ocr.core.settings import AppSettings
from verification_ocr.core.utils import calculate_war_time, extract_day_and_hour
from verification_ocr.enums import Faction
from verification_ocr.models import Verification
from verification_ocr.services.ocr.profile import (
    PROFILE_REF_HEIGHT,
    Box,
    detect_profile_boxes,
)
from verification_ocr.services.ocr.shard import detect_shard_box
from verification_ocr.services.war_service import get_war_service

# Minimum template match confidence for faction detection
FACTION_MATCH_THRESHOLD = 0.5

logger = logging.getLogger(__name__)

# Minimum image size for processing
MIN_IMAGE_DIMENSION = 100


def _get_current_ingame_time() -> str | None:
    """Get the current in-game time formatted as 'day, HH:MM'.

    Returns:
        Formatted time string or None if war service not configured.
    """
    war_service = get_war_service()
    if war_service.state.start_time is None:
        return None

    day, hour, minute = calculate_war_time(start_time=war_service.state.start_time)
    return f"{day}, {hour:02d}:{minute:02d}"


class ProfileData(TypedDict):
    """Profile data extracted from profile boxes."""

    name: str | None
    faction: Faction | None
    level: int | None
    regiment: str | None


class ShardData(TypedDict):
    """Shard data extracted from shard box."""

    shard: str | None
    ingame_time: str | None


class OCRService:
    """Service for OCR text extraction and image processing."""

    def __init__(self, settings: AppSettings, debug_mode: bool = False) -> None:
        """Initialize the OCR service.

        Args:
            settings: Application settings.
            debug_mode: Enable debug output.
        """
        self.settings = settings
        self.debug_mode = debug_mode

        if settings.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.ocr.tesseract_cmd

        self.colonial_icon = self._load_icon(path=settings.ocr.colonial_icon_path, name="colonial")
        self.warden_icon = self._load_icon(path=settings.ocr.warden_icon_path, name="warden")

        if settings.ocr.debug_mode:
            os.makedirs(name=settings.ocr.debug_output_dir, exist_ok=True)

    def _load_icon(self, path: str | None, name: str) -> MatLike | None:
        """Load a faction icon from path.

        Args:
            path: Path to icon file.
            name: Icon name for logging.

        Returns:
            Loaded icon image or None if path is empty or load fails.
        """
        if not path:
            return None
        icon = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
        if icon is None:
            logger.warning(f"Failed to load {name} icon from {path}")
        return icon

    def _validate_image(self, image_bytes: bytes) -> MatLike:
        """Decode and validate an image.

        Args:
            image_bytes: Raw image bytes to decode.

        Returns:
            Decoded BGR image.

        Raises:
            ValueError: If image cannot be decoded or is too small.
        """
        nparr = np.frombuffer(buffer=image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        height, width = image.shape[:2]
        if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
            raise ValueError(
                f"Image too small ({width}x{height}), "
                f"minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"
            )

        return image

    def detect_profile_boxes(self, image: MatLike) -> list[Box] | None:
        """Detect profile boxes in an image.

        Args:
            image: BGR image to analyze.

        Returns:
            List of 4 boxes [Username, Icon, Level, Regiment] or None.
        """
        return detect_profile_boxes(image)

    def detect_shard_box(self, image: MatLike, profile_height: int) -> Box | None:
        """Detect shard box in an image.

        Args:
            image: BGR image to analyze.
            profile_height: Height of a profile box (for scaling).

        Returns:
            Box (x, y, w, h) for the shard, or None if not found.
        """
        return detect_shard_box(image, profile_height)

    def _crop_box(self, image: MatLike, box: Box) -> MatLike:
        """Crop a region from an image.

        Args:
            image: Source BGR image.
            box: Box coordinates (x, y, w, h).

        Returns:
            Cropped image region.
        """
        x, y, w, h = box
        return image[y : y + h, x : x + w]

    def _preprocess_for_ocr(self, image: MatLike) -> MatLike:
        """Preprocess image for OCR.

        Args:
            image: BGR image to preprocess.

        Returns:
            Grayscale image.
        """
        return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    def _extract_text(self, image: MatLike) -> str:
        """Extract text from an image region using OCR.

        Args:
            image: BGR image to extract text from.

        Returns:
            Extracted text, stripped of whitespace.
        """
        preprocessed = self._preprocess_for_ocr(image)
        text: str = pytesseract.image_to_string(
            image=preprocessed,
            config="--psm 7",  # Single line mode
        )
        return text.strip()

    def _extract_level(self, image: MatLike) -> int | None:
        """Extract level number from an image region.

        Args:
            image: BGR image containing level text.

        Returns:
            Level as integer, or None if extraction fails.
        """
        preprocessed = self._preprocess_for_ocr(image)
        text = pytesseract.image_to_string(
            image=preprocessed,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        )
        text = text.strip()
        if text.isdigit():
            return int(text)
        return None

    def _extract_regiment(self, image: MatLike) -> str | None:
        """Extract regiment name from an image region.

        Expected format: [TAG#NUMBERS] Regiment Name (NUMBERS TEXT)
        Returns the text before '(' if valid regiment pattern found.

        Args:
            image: BGR image containing regiment text.

        Returns:
            Regiment name (e.g. "[7-HP#123] 7th Hispanic Platoon") or None.
        """
        preprocessed = self._preprocess_for_ocr(image)
        text = pytesseract.image_to_string(
            image=preprocessed,
            config="--psm 7",
        )
        text = text.strip()

        # Validate regiment format: must contain [, #, ]
        if not re.search(r"\[.*#.*\]", text):
            return None

        # Extract text before '(' if present
        if "(" in text:
            text = text.split("(")[0].strip()

        return text if text else None

    def _extract_shard_data(self, image: MatLike, box: Box) -> ShardData:
        """Extract shard data from shard box (multiline text).

        Expected format (3+ lines):
            Line 1: Location name (ignored)
            Line 2: "Day NNN, HHMM Hours" -> extract day and time (language-agnostic)
            Line 3: Shard name (e.g., "Able", "Baker", "Charlie")

        Args:
            image: BGR image containing the shard box.
            box: Shard box coordinates (x, y, w, h).

        Returns:
            Dictionary with keys: shard, ingame_time.
        """
        crop = self._crop_box(image=image, box=box)
        preprocessed = self._preprocess_for_ocr(crop)

        # Use PSM 6 for multiline text block
        text = pytesseract.image_to_string(
            image=preprocessed,
            config="--psm 6",
        )

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        shard: str | None = None
        ingame_time: str | None = None

        # Find the line with time pattern (digits + comma + 4 digits)
        # Shard name is on the next line
        for i, line in enumerate(lines):
            # Look for pattern: digits + comma + 4 digits (e.g., "154, 2335")
            if re.search(r"\d{1,4}\s*,\s*\d{4}", line):
                # Use language-agnostic extraction
                found_time = extract_day_and_hour(line)
                if found_time and "," in found_time and ":" in found_time:
                    ingame_time = found_time
                    # Shard is on the next line
                    if i + 1 < len(lines):
                        shard = lines[i + 1].upper().strip()
                    break

        return ShardData(
            shard=shard,
            ingame_time=ingame_time,
        )

    def _detect_faction(self, image: MatLike) -> Faction | None:
        """Detect faction from icon using template matching.

        Scales the square icon templates to fit the smaller dimension of the
        icon region, then performs template matching.

        Args:
            image: BGR image of the icon box.

        Returns:
            Detected Faction or None if no match found.
        """
        if self.colonial_icon is None and self.warden_icon is None:
            return None

        # Icon is square, sized to fit the region (use smaller dimension)
        h, w = image.shape[:2]
        icon_size = min(h, w)

        best_faction: Faction | None = None
        best_score = FACTION_MATCH_THRESHOLD

        for faction, template in [
            (Faction.COLONIAL, self.colonial_icon),
            (Faction.WARDEN, self.warden_icon),
        ]:
            if template is None:
                continue

            # Scale template to square size
            scaled = cv2.resize(
                src=template,
                dsize=(icon_size, icon_size),
                interpolation=cv2.INTER_CUBIC,
            )

            # Template must fit in the icon region
            if scaled.shape[0] > h or scaled.shape[1] > w:
                continue

            # Template match
            result = cv2.matchTemplate(
                image=image,
                templ=scaled,
                method=cv2.TM_CCOEFF_NORMED,
            )
            _, max_val, _, _ = cv2.minMaxLoc(src=result)

            if max_val > best_score:
                best_score = max_val
                best_faction = faction

        return best_faction

    def extract_profile_data(
        self,
        image: MatLike,
        boxes: list[Box],
    ) -> ProfileData:
        """Extract profile data from detected boxes.

        Args:
            image: BGR image containing the profile.
            boxes: List of 4 boxes [Username, Icon, Level, Regiment].

        Returns:
            Dictionary with keys: name, faction, level, regiment.
        """
        username_box, icon_box, level_box, regiment_box = boxes

        # Scale all crops if box height is below reference
        box_height = username_box[3]
        scale_factor = PROFILE_REF_HEIGHT / box_height if box_height < PROFILE_REF_HEIGHT else 1.0

        def crop_and_scale(box: Box) -> MatLike:
            crop = self._crop_box(image=image, box=box)
            if scale_factor > 1.0:
                return cv2.resize(
                    src=crop,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )
            return crop

        # Extract data from each box
        name = self._extract_text(crop_and_scale(username_box))
        faction = self._detect_faction(crop_and_scale(icon_box))
        level = self._extract_level(crop_and_scale(level_box))
        regiment = self._extract_regiment(crop_and_scale(regiment_box))

        return ProfileData(
            name=name or None,
            faction=faction,
            level=level,
            regiment=regiment or None,
        )

    def verify(self, image1_bytes: bytes, image2_bytes: bytes) -> Verification:
        """Process two images and extract user verification information.

        Detects profile boxes in one image and shard box in the other.
        Profile detection is tried on image1 first, then image2 if it fails.

        Args:
            image1_bytes: First image bytes.
            image2_bytes: Second image bytes.

        Returns:
            Verification result with extracted data.

        Raises:
            ValueError: If images cannot be decoded, are invalid, or
                required boxes cannot be detected.
        """
        image1 = self._validate_image(image1_bytes)
        image2 = self._validate_image(image2_bytes)

        # Try profile detection on image1 first, then image2
        profile_boxes = self.detect_profile_boxes(image1)
        profile_image = image1
        shard_image = image2

        if profile_boxes is None:
            profile_boxes = self.detect_profile_boxes(image2)
            profile_image = image2
            shard_image = image1

        if profile_boxes is None:
            raise ValueError("Could not detect profile boxes in either image")

        # Get profile height for shard detection scaling
        profile_height = profile_boxes[0][3]

        # Detect shard box in the other image
        shard_box = self.detect_shard_box(shard_image, profile_height)
        if shard_box is None:
            raise ValueError("Could not detect shard box")

        # Extract profile data
        profile_data = self.extract_profile_data(
            image=profile_image,
            boxes=profile_boxes,
        )

        # Extract shard data
        shard_data = self._extract_shard_data(
            image=shard_image,
            box=shard_box,
        )

        # Get war info
        war_service = get_war_service()

        return Verification(
            name=profile_data["name"],
            level=profile_data["level"],
            regiment=profile_data["regiment"],
            faction=profile_data["faction"],
            shard=shard_data["shard"],
            ingame_time=shard_data["ingame_time"],
            war_number=war_service.state.war_number,
            current_ingame_time=_get_current_ingame_time(),
        )
