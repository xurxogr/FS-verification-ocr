"""Tests for settings module initialization."""

from verification_ocr.core.settings import AppSettings, get_settings, reload_settings


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_app_settings(self) -> None:
        """
        Test that get_settings returns AppSettings instance.

        Returns:
            None
        """
        settings = get_settings()
        assert isinstance(settings, AppSettings)

    def test_get_settings_is_cached(self) -> None:
        """
        Test that get_settings returns the same cached instance.

        Returns:
            None
        """
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_has_api_server(self) -> None:
        """
        Test that settings has api_server attribute.

        Returns:
            None
        """
        settings = get_settings()
        assert hasattr(settings, "api_server")

    def test_get_settings_has_ocr(self) -> None:
        """
        Test that settings has ocr attribute.

        Returns:
            None
        """
        settings = get_settings()
        assert hasattr(settings, "ocr")

    def test_get_settings_has_logging(self) -> None:
        """
        Test that settings has logging attribute.

        Returns:
            None
        """
        settings = get_settings()
        assert hasattr(settings, "logging")


class TestReloadSettings:
    """Tests for reload_settings function."""

    def test_reload_settings_returns_app_settings(self) -> None:
        """
        Test that reload_settings returns AppSettings instance.

        Returns:
            None
        """
        settings = reload_settings()
        assert isinstance(settings, AppSettings)

    def test_reload_settings_clears_cache(self) -> None:
        """
        Test that reload_settings clears the cache.

        Returns:
            None
        """
        settings1 = get_settings()
        settings2 = reload_settings()
        # After reload, should be a new instance
        assert settings1 is not settings2

    def test_reload_settings_new_instance_on_subsequent_get(self) -> None:
        """
        Test that get_settings returns new instance after reload.

        Returns:
            None
        """
        settings1 = get_settings()
        reload_settings()
        settings2 = get_settings()
        # settings2 should be the reloaded instance, not settings1
        assert settings1 is not settings2
