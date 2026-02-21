"""Verification service - OCR processing of image pairs."""

import logging
import os
import re

import cv2
import numpy as np
import pytesseract
from cv2.typing import MatLike

from verification_ocr.core.settings import AppSettings
from verification_ocr.core.utils import calculate_war_time
from verification_ocr.models import ImageRegions, Region, Verification, VerificationResponse
from verification_ocr.services.war_service import get_war_service

logger = logging.getLogger(__name__)

# Threshold for binary image conversion (same as foxhole_stockpiles)
TESSERACT_BINARY_THRESHOLD = 127


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
        time_parts = parts[1].split(":")
        if len(time_parts) != 2:
            return None

        hour = int(time_parts[0])
        minute = int(time_parts[1])

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

        # Load colonial icon if path is provided
        self.colonial_icon: cv2.typing.MatLike | None = None
        if settings.ocr.colonial_icon_path:
            self.colonial_icon = cv2.imread(settings.ocr.colonial_icon_path, cv2.IMREAD_COLOR)
            if self.colonial_icon is None:
                logger.warning(
                    f"Failed to load colonial icon from {settings.ocr.colonial_icon_path}"
                )

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
        text = pytesseract.image_to_string(
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

        Uses foxhole_stockpiles preprocessing logic with dynamic thresholding.

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

        # Calculate threshold from this region's pixel distribution
        unique_values, counts = np.unique(upscaled, return_counts=True)
        most_common_value = unique_values[np.argmax(counts)]
        threshold_value = most_common_value + 120 * (1 - most_common_value / 255)

        # For shard: use THRESH_BINARY (not inverted) and reduce threshold
        threshold_mode = cv2.THRESH_BINARY
        threshold_value -= 30

        # Zero out pixels below threshold
        upscaled[upscaled < threshold_value] = 0

        # Apply binary threshold
        _, binary = cv2.threshold(
            src=upscaled,
            thresh=TESSERACT_BINARY_THRESHOLD,
            maxval=255,
            type=threshold_mode,
        )

        # Dilate to connect text components
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        binary = cv2.dilate(src=binary, kernel=kernel, iterations=1)

        # Convert back to RGB for tesseract
        result = cv2.cvtColor(src=binary, code=cv2.COLOR_GRAY2RGB)

        return np.asarray(a=result, dtype=np.uint8)

    def _find_colonial_icon(self, image: cv2.typing.MatLike) -> bool | None:
        """
        Find the colonial icon in the image using template matching.

        Args:
            image (cv2.typing.MatLike): Image to search for colonial icon.

        Returns:
            bool | None: True if colonial icon found, False if not, None if no template.
        """
        if self.colonial_icon is None:
            return None

        # Calculate the scale to match the height of the image
        scale = image.shape[0] / self.colonial_icon.shape[0]

        # Resize the template according to the scale
        resized_template = cv2.resize(
            src=self.colonial_icon,
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
        return max_val > 0.7

    def _calculate_regions(
        self,
        image: cv2.typing.MatLike,
    ) -> ImageRegions:
        """
        Calculate region coordinates based on image dimensions.

        Uses foxhole_stockpiles scaling logic with 4K (2160p) as base resolution.

        Args:
            image (cv2.typing.MatLike): Image to calculate regions for.

        Returns:
            ImageRegions: Calculated region coordinates.
        """
        height, width = image.shape[:2]
        scale_factor = height / self.settings.ocr.base_height

        box_width = int(self.settings.ocr.base_box_width * scale_factor)
        box_height = int(self.settings.ocr.base_box_height * scale_factor)

        # Profile regions (using proportional coordinates)
        py = height / 5
        px = width / 5

        # Shard region (bottom-left corner, same logic as foxhole_stockpiles)
        shard_x = box_height
        shard_y = height - int(self.settings.ocr.base_box_height * scale_factor * 3)
        shard_width = int(box_width * 3.5)
        shard_height = box_height

        # Common y coordinates for username/icon/level row
        row_y1 = int(0.62 * py)
        row_y2 = int(0.775 * py)

        return ImageRegions(
            username=Region(
                y1=row_y1,
                y2=row_y2,
                x1=int(1.81 * px),
                x2=int(2.25 * px),
            ),
            icon=Region(
                y1=row_y1,
                y2=row_y2,
                x1=int(2.34 * px),
                x2=int(2.49 * px),
            ),
            level=Region(
                y1=row_y1,
                y2=row_y2,
                x1=int(2.51 * px),
                x2=int(2.83 * px),
            ),
            regiment=Region(
                y1=int(1.22 * py),
                y2=int(1.34 * py),
                x1=int(2.42 * px),
                x2=int(3.5 * px),
            ),
            shard=Region(
                y1=shard_y,
                y2=shard_y + shard_height,
                x1=shard_x,
                x2=shard_x + shard_width,
            ),
            scale_factor=scale_factor,
        )

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
            filename (str): Output filename.
        """
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

        output_path = os.path.join(self.settings.ocr.debug_output_dir, filename)
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

        # Extract username from dedicated region
        r = regions.username
        username_image = image[r.y1 : r.y2, r.x1 : r.x2]
        username_text = self._extract_text_from_image(username_image)
        data.name = username_text.strip() if username_text.strip() else None

        # No name found - either image is too small or this is a map image
        if data.name is None:
            return data

        # Extract level from dedicated region (language-agnostic: look for : and digits)
        r = regions.level
        level_image = image[r.y1 : r.y2, r.x1 : r.x2]
        level_text = self._extract_text_from_image(level_image)
        # Look for colon and extract digits after it
        if ":" in level_text:
            level_part = level_text.split(":")[-1]
            digits = "".join(c for c in level_part if c.isdigit())
            if digits:
                data.level = int(digits)

        # Check for colonial faction icon in icon region
        r = regions.icon
        icon_image = image[r.y1 : r.y2, r.x1 : r.x2]
        data.colonial = self._find_colonial_icon(icon_image)

        # Extract Regiment name from grey bar region
        r = regions.regiment
        regiment_image = image[r.y1 : r.y2, r.x1 : r.x2]
        regiment_text = self._extract_text_from_image(regiment_image, scale=True)
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

    def verify(self, image1_bytes: bytes, image2_bytes: bytes) -> VerificationResponse:
        """
        Process two images and extract user verification information.

        Args:
            image1_bytes (bytes): First image bytes.
            image2_bytes (bytes): Second image bytes.

        Returns:
            VerificationResponse: Verification result with user info or error.
        """
        logger.info("Processing image pair for verification")

        # Decode images
        images: list[MatLike | None] = []
        for img_bytes in [image1_bytes, image2_bytes]:
            nparr = np.frombuffer(buffer=img_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
            images.append(img)

        # Check if images were decoded successfully
        if images[0] is None or images[1] is None:
            return VerificationResponse(
                success=False,
                error="Failed to decode one or both images",
            )

        # Calculate regions for both images
        regions1 = self._calculate_regions(images[0])
        regions2 = self._calculate_regions(images[1])

        # Save debug images if debug mode is enabled
        if self.settings.ocr.debug_mode:
            self._save_debug_image(
                image=images[0],
                regions=regions1,
                filename="debug_image1_regions.png",
            )
            self._save_debug_image(
                image=images[1],
                regions=regions2,
                filename="debug_image2_regions.png",
            )

        # Try to find user info in first image, fallback to second
        verification = self._find_user_info(image=images[0], regions=regions1)
        if verification.name is None:
            verification = self._find_user_info(image=images[1], regions=regions2)
            shard, ingame_time = self._get_shard_and_time(
                image=images[0],
                regions=regions1,
            )
        else:
            shard, ingame_time = self._get_shard_and_time(
                image=images[1],
                regions=regions2,
            )

        verification.shard = shard
        verification.ingame_time = ingame_time

        if verification.name is None:
            return VerificationResponse(
                success=False,
                error="No name found in any of the images",
            )

        # Validate in-game time difference
        if ingame_time is not None:
            parsed_time = parse_ingame_time(ingame_time=ingame_time)
            current_time = get_current_ingame_time()

            if parsed_time is not None and current_time is not None:
                time_diff = calculate_ingame_time_diff(
                    extracted_day=parsed_time[0],
                    extracted_hour=parsed_time[1],
                    current_day=current_time[0],
                    current_hour=current_time[1],
                )

                max_diff = self.settings.verification.max_ingame_time_diff
                if time_diff > max_diff:
                    return VerificationResponse(
                        success=False,
                        error=(
                            f"In-game time difference is {time_diff} hours "
                            f"(max allowed: {max_diff})"
                        ),
                        verification=verification,
                    )

        return VerificationResponse(
            success=True,
            verification=verification,
        )
