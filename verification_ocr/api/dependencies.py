"""Dependency injection providers."""

from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from verification_ocr.core.settings import get_settings
from verification_ocr.services.verification_service import VerificationService

# API key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache
def get_verification_service() -> VerificationService:
    """
    Get cached verification service singleton.

        VerificationService: The verification service instance.
    """
    settings = get_settings()
    return VerificationService(settings)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header.

    Returns:
        str | None: The validated API key, or None if auth is disabled.

    Raises:
        HTTPException: 401 if API key is required but missing/invalid.
    """
    settings = get_settings()
    configured_key = settings.api_server.api_key

    # If no API key is configured, auth is disabled
    if configured_key is None:
        return None

    # API key is required but not provided
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # API key doesn't match
    if api_key != configured_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def clear_dependency_caches() -> None:
    """
    Clear all dependency caches.

    """
    get_verification_service.cache_clear()
