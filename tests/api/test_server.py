"""Tests for API server."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from verification_ocr import __version__
from verification_ocr.api.server import (
    app,
    create_app,
    lifespan,
    rate_limit_exceeded_handler,
)
from verification_ocr.services import get_war_service


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_success(self, mock_tesseract_available: MagicMock) -> None:
        """Test lifespan with tesseract available."""
        mock_app = MagicMock(spec=FastAPI)

        async with lifespan(mock_app):
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_lifespan_no_tesseract_raises(
        self,
        mock_tesseract_unavailable: MagicMock,
    ) -> None:
        """Test lifespan raises when tesseract unavailable."""
        mock_app = MagicMock(spec=FastAPI)

        with pytest.raises(RuntimeError) as exc_info:
            async with lifespan(mock_app):
                pass

        assert "Tesseract OCR is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_lifespan_skips_api_fetch_when_war_configured(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test lifespan skips API fetch when war state is already configured."""
        from verification_ocr.core.settings import get_settings

        war_service = get_war_service()
        mock_app = MagicMock(spec=FastAPI)

        # Patch settings to have configured war state
        settings = get_settings()
        with patch.object(settings.war, "number", 132):
            with patch.object(settings.war, "start_time", 1770663602746):
                with patch.object(
                    war_service, "sync_from_api", new_callable=AsyncMock
                ) as mock_sync:
                    async with lifespan(mock_app):
                        # API sync should NOT be called when war is already configured
                        mock_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_fetches_from_api_when_war_not_configured(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test lifespan fetches war data from API when not configured."""
        from verification_ocr.core.settings import get_settings

        war_service = get_war_service()

        mock_app = MagicMock(spec=FastAPI)

        # Patch settings to return unconfigured war state
        settings = get_settings()
        with patch.object(settings.war, "number", None):
            with patch.object(settings.war, "start_time", None):
                with patch.object(
                    war_service, "sync_from_api", new_callable=AsyncMock
                ) as mock_sync:
                    mock_sync.return_value = True

                    async with lifespan(mock_app):
                        mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_logs_warning_when_war_state_not_configured_after_sync(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test lifespan logs warning when war state cannot be configured."""
        from verification_ocr.core.settings import get_settings

        war_service = get_war_service()

        mock_app = MagicMock(spec=FastAPI)

        # Patch settings to return unconfigured war state
        settings = get_settings()
        with patch.object(settings.war, "number", None):
            with patch.object(settings.war, "start_time", None):
                with patch.object(
                    war_service, "sync_from_api", new_callable=AsyncMock
                ) as mock_sync:
                    mock_sync.return_value = False

                    with patch("verification_ocr.api.server.logger.warning") as mock_warning:
                        async with lifespan(mock_app):
                            mock_warning.assert_called_once_with(
                                "War state not configured and could not be fetched from API"
                            )


class TestCreateApp:
    """Tests for create_app function."""

    def test_returns_fastapi_instance(self) -> None:
        """Test that create_app returns FastAPI instance."""
        app_instance = create_app()
        assert isinstance(app_instance, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """Test that app has correct title."""
        app_instance = create_app()
        assert app_instance.title == "Verification OCR Service"

    def test_app_has_correct_version(self) -> None:
        """Test that app has correct version."""
        app_instance = create_app()
        assert app_instance.version == __version__

    def test_create_app_without_static_dir(self) -> None:
        """Test create_app when static directory does not exist."""
        with patch("verification_ocr.api.server.os.path.exists", return_value=False):
            app_instance = create_app()
            assert isinstance(app_instance, FastAPI)
            # Static route should not be mounted
            routes = [route.path for route in app_instance.routes]
            assert "/static" not in routes


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_healthy(self, test_client: TestClient) -> None:
        """Test that health check returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_returns_version(self, test_client: TestClient) -> None:
        """Test that health check returns version."""
        response = test_client.get("/health")
        data = response.json()
        assert data["version"] == __version__

    def test_health_check_returns_tesseract_version(
        self,
        test_client: TestClient,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test that health check returns tesseract version."""
        response = test_client.get("/health")
        data = response.json()
        assert "tesseract_version" in data


class TestWarEndpoint:
    """Tests for /war endpoint."""

    def test_war_info_returns_200(self, test_client: TestClient) -> None:
        """Test that /war returns 200 status code."""
        response = test_client.get("/war")
        assert response.status_code == 200

    def test_war_info_returns_none_when_not_configured(self, test_client: TestClient) -> None:
        """Test that /war returns None values when not configured."""
        war_service = get_war_service()
        original_number = war_service.state.war_number
        original_time = war_service.state.start_time

        war_service.initialize(war_number=None, start_time=None)

        try:
            response = test_client.get("/war")
            data = response.json()
            assert data["war_number"] is None
            assert data["war_day"] is None
            assert data["war_hour"] is None
            assert data["war_minute"] is None
            assert data["start_time"] is None
        finally:
            war_service.initialize(war_number=original_number, start_time=original_time)

    def test_war_info_returns_configured_values(self, test_client: TestClient) -> None:
        """Test that /war returns configured war values."""
        war_service = get_war_service()
        original_number = war_service.state.war_number
        original_time = war_service.state.start_time

        war_service.initialize(war_number=132, start_time=1770663602746)

        try:
            response = test_client.get("/war")
            data = response.json()
            assert data["war_number"] == 132
            assert data["start_time"] == 1770663602746
            assert data["war_day"] is not None
            assert data["war_hour"] is not None
            assert data["war_minute"] is not None
        finally:
            war_service.initialize(war_number=original_number, start_time=original_time)

    def test_war_day_calculation(self, test_client: TestClient) -> None:
        """Test that war_day is calculated correctly."""
        import time

        war_service = get_war_service()
        original_number = war_service.state.war_number
        original_time = war_service.state.start_time

        # Set start time to 24 hours ago
        war_service.initialize(
            war_number=132,
            start_time=int((time.time() - 24 * 60 * 60) * 1000),
        )

        try:
            response = test_client.get("/war")
            data = response.json()
            # Should be approximately 24 hours (Day 25, since game starts at Day 1)
            assert data["war_day"] == 25
        finally:
            war_service.initialize(war_number=original_number, start_time=original_time)

    def test_war_time_calculation(self, test_client: TestClient) -> None:
        """Test that war_hour and war_minute are calculated correctly."""
        import time

        war_service = get_war_service()
        original_number = war_service.state.war_number
        original_time = war_service.state.start_time

        # Set start time to 2.5 hours ago (2 days, 12:00 in-game)
        war_service.initialize(
            war_number=132,
            start_time=int((time.time() - 2.5 * 60 * 60) * 1000),
        )

        try:
            response = test_client.get("/war")
            data = response.json()
            # 2.5 real hours = Day 3 (2+1), 12:00 in-game
            assert data["war_day"] == 3
            assert data["war_hour"] == 12
            assert data["war_minute"] == 0
        finally:
            war_service.initialize(war_number=original_number, start_time=original_time)


class TestSyncEndpoint:
    """Tests for /sync endpoint."""

    def test_sync_returns_200(self, test_client: TestClient) -> None:
        """Test that /sync returns 200 status code."""
        war_service = get_war_service()
        with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = True
            response = test_client.post("/sync")
            assert response.status_code == 200

    def test_sync_returns_success_true_on_success(self, test_client: TestClient) -> None:
        """Test that /sync returns success true when API call succeeds."""
        war_service = get_war_service()
        with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = True
            response = test_client.post("/sync")
            data = response.json()
            assert data["success"] is True

    def test_sync_returns_success_false_on_failure(self, test_client: TestClient) -> None:
        """Test that /sync returns success false when API call fails."""
        war_service = get_war_service()
        with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = False
            response = test_client.post("/sync")
            data = response.json()
            assert data["success"] is False

    def test_sync_returns_war_state(self, test_client: TestClient) -> None:
        """Test that /sync returns current war state with calculated time."""
        import time

        war_service = get_war_service()
        original_number = war_service.state.war_number
        original_time = war_service.state.start_time

        # Set start time to 2.5 hours ago
        war_service.initialize(
            war_number=132,
            start_time=int((time.time() - 2.5 * 60 * 60) * 1000),
        )

        try:
            with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = True
                response = test_client.post("/sync")
                data = response.json()
                assert data["war_number"] == 132
                assert data["start_time"] is not None
                assert data["war_day"] == 3
                assert data["war_hour"] == 12
                assert data["war_minute"] == 0
        finally:
            war_service.initialize(war_number=original_number, start_time=original_time)


class TestIndexEndpoint:
    """Tests for / endpoint."""

    def test_index_returns_response(self, test_client: TestClient) -> None:
        """Test that index returns a response."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_index_returns_json_when_no_index_html(self, test_client: TestClient) -> None:
        """Test that index returns JSON fallback when index.html doesn't exist."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        with patch.object(settings.api_server, "serve_frontend", True):
            with patch("verification_ocr.api.server.os.path.exists", return_value=False):
                response = test_client.get("/")
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Verification OCR Service"
                assert data["docs"] == "/docs"


class TestVerifyEndpoint:
    """Tests for /verify endpoint."""

    def _create_test_image(self, width: int = 3840, height: int = 2160) -> bytes:
        """Create a test image of specified size."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def test_verify_with_valid_images(self, test_client: TestClient) -> None:
        """Test verify endpoint with valid images."""
        image_bytes = self._create_test_image()

        call_count = [0]

        def mock_ocr(*args, **kwargs) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "TestUser"  # username
            if call_count[0] == 2:
                return "Level: 15"  # level
            if call_count[0] == 3:
                return ""  # regiment
            return "100, 1200\nABLE"  # shard

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            side_effect=mock_ocr,
        ):
            response = test_client.post(
                "/verify",
                files={
                    "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
                    "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["verification"]["name"] == "TestUser"

    def test_verify_missing_image1(self, test_client: TestClient) -> None:
        """Test verify endpoint with missing image1."""
        image_bytes = self._create_test_image()
        response = test_client.post(
            "/verify",
            files={
                "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
            },
        )

        assert response.status_code == 422

    def test_verify_missing_image2(self, test_client: TestClient) -> None:
        """Test verify endpoint with missing image2."""
        image_bytes = self._create_test_image()
        response = test_client.post(
            "/verify",
            files={
                "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
            },
        )

        assert response.status_code == 422

    def test_verify_returns_error_when_no_name_found(self, test_client: TestClient) -> None:
        """Test that verify returns error when no name found."""
        image_bytes = self._create_test_image()

        with patch(
            "verification_ocr.services.verification_service.pytesseract.image_to_string",
            return_value="",
        ):
            response = test_client.post(
                "/verify",
                files={
                    "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
                    "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
                },
            )

            data = response.json()
            assert data["success"] is False
            assert "No name found" in data["error"]


class TestAppInstance:
    """Tests for the app instance."""

    def test_app_is_fastapi(self) -> None:
        """Test that app is a FastAPI instance."""
        assert isinstance(app, FastAPI)

    def test_app_has_health_route(self) -> None:
        """Test that app has /health route."""
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_app_has_verify_route(self) -> None:
        """Test that app has /verify route."""
        routes = [route.path for route in app.routes]
        assert "/verify" in routes

    def test_app_has_index_route(self) -> None:
        """Test that app has / route."""
        routes = [route.path for route in app.routes]
        assert "/" in routes

    def test_app_has_war_route(self) -> None:
        """Test that app has /war route."""
        routes = [route.path for route in app.routes]
        assert "/war" in routes

    def test_app_has_sync_route(self) -> None:
        """Test that app has /sync route."""
        routes = [route.path for route in app.routes]
        assert "/sync" in routes


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_security_headers_added_to_response(self, test_client: TestClient) -> None:
        """Test that security headers are added to all responses."""
        response = test_client.get("/health")

        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "strict-origin-when-cross-origin" in response.headers.get("Referrer-Policy", "")

    def test_hsts_header_added_when_https_proxy(self, test_client: TestClient) -> None:
        """Test that HSTS header is added when behind HTTPS proxy."""
        response = test_client.get(
            "/health",
            headers={"X-Forwarded-Proto": "https"},
        )

        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers.get("Strict-Transport-Security", "")

    def test_no_hsts_header_without_https_proxy(self, test_client: TestClient) -> None:
        """Test that HSTS header is not added without HTTPS proxy."""
        response = test_client.get("/health")

        assert "Strict-Transport-Security" not in response.headers


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded_handler(self) -> None:
        """Test rate limit exceeded handler returns proper response."""
        from slowapi.errors import RateLimitExceeded

        mock_request = MagicMock()
        # RateLimitExceeded requires a Limit object, use MagicMock
        mock_limit = MagicMock()
        mock_exc = RateLimitExceeded(mock_limit)

        response = rate_limit_exceeded_handler(mock_request, mock_exc)

        assert response.status_code == 429
        assert b"Rate limit exceeded" in response.body


class TestUploadSizeLimits:
    """Tests for file upload size limits."""

    def _create_large_image(self, size_mb: int = 60) -> bytes:
        """Create a large image that exceeds default size limit."""
        # Create uncompressed data larger than the limit
        # Using a simple approach - create large random data
        return b"0" * (size_mb * 1024 * 1024)

    def test_verify_rejects_oversized_image1(self, test_client: TestClient) -> None:
        """Test that oversized first image is rejected."""
        large_data = self._create_large_image(60)  # 60MB exceeds 50MB default

        # Create a small valid image for image2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        small_bytes = buffer.tobytes()

        response = test_client.post(
            "/verify",
            files={
                "image1": ("large.png", io.BytesIO(large_data), "image/png"),
                "image2": ("small.png", io.BytesIO(small_bytes), "image/png"),
            },
        )

        assert response.status_code == 413
        assert "Image 1 exceeds maximum size" in response.json()["detail"]

    def test_verify_rejects_oversized_image2(self, test_client: TestClient) -> None:
        """Test that oversized second image is rejected."""
        large_data = self._create_large_image(60)  # 60MB exceeds 50MB default

        # Create a small valid image for image1
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        small_bytes = buffer.tobytes()

        response = test_client.post(
            "/verify",
            files={
                "image1": ("small.png", io.BytesIO(small_bytes), "image/png"),
                "image2": ("large.png", io.BytesIO(large_data), "image/png"),
            },
        )

        assert response.status_code == 413
        assert "Image 2 exceeds maximum size" in response.json()["detail"]


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_headers_when_origins_configured(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test that CORS headers are added when origins are configured."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        # Configure CORS origins
        with patch.object(
            settings.api_server,
            "cors_allow_origins",
            ["http://localhost:3000"],
        ):
            # Create a new app with the patched settings
            new_app = create_app()
            client = TestClient(new_app)

            response = client.options(
                "/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )

            # CORS preflight should work
            assert response.status_code in [200, 405]


class TestApiKeyAuthentication:
    """Tests for API key authentication."""

    def _create_test_image(self) -> bytes:
        """Create a test image."""
        img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def test_verify_api_key_disabled(self, test_client: TestClient) -> None:
        """Test that API key is not required when disabled."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        # Ensure API key is disabled (None)
        with patch.object(settings.api_server, "api_key", None):
            war_service = get_war_service()
            with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = True
                response = test_client.post("/sync")
                assert response.status_code == 200

    def test_verify_api_key_required_but_missing(self, test_client: TestClient) -> None:
        """Test that 401 is returned when API key required but not provided."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        # Enable API key requirement
        with patch.object(settings.api_server, "api_key", "test-secret-key"):
            response = test_client.post("/sync")
            assert response.status_code == 401
            assert response.json()["detail"] == "API key required"

    def test_verify_api_key_invalid(self, test_client: TestClient) -> None:
        """Test that 401 is returned when API key is invalid."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        # Enable API key requirement
        with patch.object(settings.api_server, "api_key", "test-secret-key"):
            response = test_client.post(
                "/sync",
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401
            assert response.json()["detail"] == "Invalid API key"

    def test_verify_api_key_valid(self, test_client: TestClient) -> None:
        """Test that request succeeds with valid API key."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        # Enable API key requirement
        with patch.object(settings.api_server, "api_key", "test-secret-key"):
            war_service = get_war_service()
            with patch.object(war_service, "sync_from_api", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = True
                response = test_client.post(
                    "/sync",
                    headers={"X-API-Key": "test-secret-key"},
                )
                assert response.status_code == 200

    def test_verify_endpoint_requires_api_key(self, test_client: TestClient) -> None:
        """Test that /verify endpoint requires API key when enabled."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        image_bytes = self._create_test_image()

        # Enable API key requirement
        with patch.object(settings.api_server, "api_key", "test-secret-key"):
            response = test_client.post(
                "/verify",
                files={
                    "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
                    "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
                },
            )
            assert response.status_code == 401
            assert response.json()["detail"] == "API key required"

    def test_verify_endpoint_with_valid_api_key(self, test_client: TestClient) -> None:
        """Test that /verify endpoint works with valid API key."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        image_bytes = self._create_test_image()

        # Enable API key requirement
        with patch.object(settings.api_server, "api_key", "test-secret-key"):
            with patch(
                "verification_ocr.services.verification_service.pytesseract.image_to_string",
                return_value="TestUser",
            ):
                response = test_client.post(
                    "/verify",
                    files={
                        "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
                        "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
                    },
                    headers={"X-API-Key": "test-secret-key"},
                )
                assert response.status_code == 200


class TestFrontendServing:
    """Tests for frontend serving configuration."""

    def test_static_files_not_mounted_when_frontend_disabled(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test that static files are not mounted when serve_frontend is False."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        with patch.object(settings.api_server, "serve_frontend", False):
            with patch("verification_ocr.api.server.os.path.exists", return_value=True):
                app_instance = create_app()
                routes = [route.path for route in app_instance.routes]
                assert "/static" not in routes

    def test_static_files_mounted_when_frontend_enabled(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """Test that static files are mounted when serve_frontend is True."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        with patch.object(settings.api_server, "serve_frontend", True):
            with patch("verification_ocr.api.server.os.path.exists", return_value=True):
                app_instance = create_app()
                routes = [route.path for route in app_instance.routes]
                assert "/static" in routes

    def test_index_returns_json_when_frontend_disabled(
        self,
        test_client: TestClient,
    ) -> None:
        """Test that index returns JSON when serve_frontend is False."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()
        with patch.object(settings.api_server, "serve_frontend", False):
            response = test_client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Verification OCR Service"
            assert data["docs"] == "/docs"


class TestIndexEndpointSecurity:
    """Tests for index endpoint security features."""

    def test_index_returns_file_when_exists(
        self,
        test_client: TestClient,
        tmp_path,
    ) -> None:
        """Test that index returns FileResponse when index.html exists."""
        from verification_ocr.core.settings import get_settings

        # Create a temporary index.html
        index_file = tmp_path / "index.html"
        index_file.write_text("<html><body>Test</body></html>")

        static_dir = str(tmp_path)
        settings = get_settings()

        with patch.object(settings.api_server, "serve_frontend", True):
            with patch("verification_ocr.api.server.STATIC_DIR", static_dir):
                response = test_client.get("/")

                assert response.status_code == 200

    def test_index_path_traversal_protection(self, test_client: TestClient) -> None:
        """Test that path traversal in index endpoint is prevented."""
        from verification_ocr.core.settings import get_settings

        settings = get_settings()

        # Mock realpath to simulate path resolving outside STATIC_DIR
        def mock_realpath(path):
            if "index.html" in str(path):
                return "/etc/passwd"
            return "/safe/static"

        with patch.object(settings.api_server, "serve_frontend", True):
            with patch("verification_ocr.api.server.os.path.realpath", side_effect=mock_realpath):
                response = test_client.get("/")

                # Should return JSON fallback, not file
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Verification OCR Service"
                assert data["docs"] == "/docs"
