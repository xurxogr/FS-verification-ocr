"""Dependency injection providers."""

from functools import lru_cache

from verification_ocr.core.settings import get_settings
from verification_ocr.services.verification_service import VerificationService


@lru_cache
def get_verification_service() -> VerificationService:
    """
    Get cached verification service singleton.

    Returns:
        VerificationService: The verification service instance.
    """
    settings = get_settings()
    return VerificationService(settings)


def clear_dependency_caches() -> None:
    """
    Clear all dependency caches.

    Returns:
        None
    """
    get_verification_service.cache_clear()
