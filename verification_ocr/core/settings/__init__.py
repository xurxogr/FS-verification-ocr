"""Settings module - pydantic-settings based configuration."""

from functools import lru_cache

from verification_ocr.core.settings.app_settings import AppSettings


@lru_cache
def get_settings() -> AppSettings:
    """
    Get cached application settings singleton.

    Returns:
        AppSettings: The application settings instance.
    """
    return AppSettings()


def reload_settings() -> AppSettings:
    """
    Clear settings cache and reload.

    Returns:
        AppSettings: The newly loaded application settings instance.
    """
    get_settings.cache_clear()
    return get_settings()


__all__ = ["AppSettings", "get_settings", "reload_settings"]
