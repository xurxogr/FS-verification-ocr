"""Tests for API dependencies."""

from verification_ocr.api.dependencies import (
    clear_dependency_caches,
    get_verification_service,
)
from verification_ocr.services.verification_service import VerificationService


class TestGetVerificationService:
    """Tests for get_verification_service function."""

    def test_returns_verification_service(self) -> None:
        """
        Test that get_verification_service returns VerificationService.

        """
        clear_dependency_caches()
        service = get_verification_service()
        assert isinstance(service, VerificationService)

    def test_is_cached(self) -> None:
        """
        Test that get_verification_service returns cached instance.

        """
        clear_dependency_caches()
        service1 = get_verification_service()
        service2 = get_verification_service()
        assert service1 is service2

    def test_uses_settings(self) -> None:
        """
        Test that service uses settings from get_settings.

        """
        clear_dependency_caches()
        service = get_verification_service()
        assert service.settings is not None
        assert hasattr(service.settings, "ocr")


class TestClearDependencyCaches:
    """Tests for clear_dependency_caches function."""

    def test_clears_verification_service_cache(self) -> None:
        """
        Test that clear_dependency_caches clears the service cache.

        """
        clear_dependency_caches()
        service1 = get_verification_service()
        clear_dependency_caches()
        service2 = get_verification_service()
        # After clearing, should be a new instance
        assert service1 is not service2

    def test_no_error_when_cache_empty(self) -> None:
        """
        Test that clear_dependency_caches doesn't error on empty cache.

        """
        clear_dependency_caches()
        clear_dependency_caches()  # Should not raise
