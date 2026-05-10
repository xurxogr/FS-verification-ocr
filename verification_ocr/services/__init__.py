"""Business logic services."""

from verification_ocr.services.ocr import OCRService
from verification_ocr.services.war_service import WarService, get_war_service

__all__ = [
    "OCRService",
    "WarService",
    "get_war_service",
]
