"""Tests for API server."""

import io
from unittest.mock import MagicMock, patch

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
)


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_success(self, mock_tesseract_available: MagicMock) -> None:
        """
        Test lifespan with tesseract available.

        Args:
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        mock_app = MagicMock(spec=FastAPI)

        async with lifespan(mock_app):
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_lifespan_no_tesseract_raises(
        self,
        mock_tesseract_unavailable: MagicMock,
    ) -> None:
        """
        Test lifespan raises when tesseract unavailable.

        Args:
            mock_tesseract_unavailable (MagicMock): Mock for tesseract unavailability.

        Returns:
            None
        """
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
        """
        Test lifespan skips API fetch when war state is already configured.

        Args:
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        from verification_ocr.services import get_war_state

        state = get_war_state()
        state.war_number = 132
        state.start_time = 1770663602746

        mock_app = MagicMock(spec=FastAPI)

        with patch(
            "verification_ocr.api.server.sync_war_from_api",
        ) as mock_sync:
            async with lifespan(mock_app):
                # API sync should NOT be called when war is already configured
                mock_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_fetches_from_api_when_war_not_configured(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """
        Test lifespan fetches war data from API when not configured.

        Args:
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        from verification_ocr.services import get_war_state

        state = get_war_state()

        mock_app = MagicMock(spec=FastAPI)

        with patch(
            "verification_ocr.api.server.initialize_war_state_from_settings",
        ):
            # State remains unconfigured after initialize
            state.war_number = None
            state.start_time = None

            with patch(
                "verification_ocr.api.server.sync_war_from_api",
            ) as mock_sync:
                # Simulate successful API sync
                async def mock_sync_success():
                    state.war_number = 132
                    state.start_time = 1770663602746
                    return True

                mock_sync.side_effect = mock_sync_success

                async with lifespan(mock_app):
                    mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_logs_warning_when_war_state_not_configured_after_sync(
        self,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """
        Test lifespan logs warning when war state cannot be configured.

        Args:
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        from verification_ocr.services import get_war_state

        state = get_war_state()

        mock_app = MagicMock(spec=FastAPI)

        with patch(
            "verification_ocr.api.server.initialize_war_state_from_settings",
        ):
            # State remains unconfigured after initialize
            state.war_number = None
            state.start_time = None

            with patch(
                "verification_ocr.api.server.sync_war_from_api",
                return_value=False,
            ):
                with patch(
                    "verification_ocr.api.server.logger.warning",
                ) as mock_warning:
                    async with lifespan(mock_app):
                        # Should log warning since war state still not configured
                        mock_warning.assert_called_once_with(
                            "War state not configured and could not be fetched from API"
                        )


class TestCreateApp:
    """Tests for create_app function."""

    def test_returns_fastapi_instance(self) -> None:
        """
        Test that create_app returns FastAPI instance.

        Returns:
            None
        """
        app_instance = create_app()
        assert isinstance(app_instance, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """
        Test that app has correct title.

        Returns:
            None
        """
        app_instance = create_app()
        assert app_instance.title == "Verification OCR Service"

    def test_app_has_correct_version(self) -> None:
        """
        Test that app has correct version.

        Returns:
            None
        """
        app_instance = create_app()
        assert app_instance.version == __version__

    def test_create_app_without_static_dir(self) -> None:
        """
        Test create_app when static directory does not exist.

        Returns:
            None
        """
        with patch("verification_ocr.api.server.os.path.exists", return_value=False):
            app_instance = create_app()
            assert isinstance(app_instance, FastAPI)
            # Static route should not be mounted
            routes = [route.path for route in app_instance.routes]
            assert "/static" not in routes


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_healthy(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that health check returns healthy status.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_returns_version(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that health check returns version.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        response = test_client.get("/health")
        data = response.json()
        assert data["version"] == __version__

    def test_health_check_returns_tesseract_version(
        self,
        test_client: TestClient,
        mock_tesseract_available: MagicMock,
    ) -> None:
        """
        Test that health check returns tesseract version.

        Args:
            test_client (TestClient): Test client fixture.
            mock_tesseract_available (MagicMock): Mock for tesseract availability.

        Returns:
            None
        """
        response = test_client.get("/health")
        data = response.json()
        assert "tesseract_version" in data


class TestWarEndpoint:
    """Tests for /war endpoint."""

    def test_war_info_returns_200(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /war returns 200 status code.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        response = test_client.get("/war")
        assert response.status_code == 200

    def test_war_info_returns_none_when_not_configured(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /war returns None values when not configured.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        from verification_ocr.services import get_war_state

        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        state.war_number = None
        state.start_time = None

        try:
            response = test_client.get("/war")
            data = response.json()
            assert data["war_number"] is None
            assert data["war_day"] is None
            assert data["war_hour"] is None
            assert data["war_minute"] is None
            assert data["start_time"] is None
        finally:
            state.war_number = original_number
            state.start_time = original_time

    def test_war_info_returns_configured_values(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /war returns configured war values.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        from verification_ocr.services import get_war_state

        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        state.war_number = 132
        state.start_time = 1770663602746

        try:
            response = test_client.get("/war")
            data = response.json()
            assert data["war_number"] == 132
            assert data["start_time"] == 1770663602746
            assert data["war_day"] is not None
            assert data["war_hour"] is not None
            assert data["war_minute"] is not None
        finally:
            state.war_number = original_number
            state.start_time = original_time

    def test_war_day_calculation(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that war_day is calculated correctly.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.services import get_war_state

        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        # Set start time to 24 hours ago
        state.war_number = 132
        state.start_time = int((time.time() - 24 * 60 * 60) * 1000)

        try:
            response = test_client.get("/war")
            data = response.json()
            # Should be approximately 24 hours (Day 25, since game starts at Day 1)
            assert data["war_day"] == 25
        finally:
            state.war_number = original_number
            state.start_time = original_time

    def test_war_time_calculation(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that war_hour and war_minute are calculated correctly.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.services import get_war_state

        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        # Set start time to 2.5 hours ago (2 days, 12:00 in-game)
        state.war_number = 132
        state.start_time = int((time.time() - 2.5 * 60 * 60) * 1000)

        try:
            response = test_client.get("/war")
            data = response.json()
            # 2.5 real hours = Day 3 (2+1), 12:00 in-game
            assert data["war_day"] == 3
            assert data["war_hour"] == 12
            assert data["war_minute"] == 0
        finally:
            state.war_number = original_number
            state.start_time = original_time


class TestSyncEndpoint:
    """Tests for /sync endpoint."""

    def test_sync_returns_200(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /sync returns 200 status code.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        with patch(
            "verification_ocr.api.server.sync_war_from_api",
            return_value=True,
        ):
            response = test_client.post("/sync")
            assert response.status_code == 200

    def test_sync_returns_success_true_on_success(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /sync returns success true when API call succeeds.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        with patch(
            "verification_ocr.api.server.sync_war_from_api",
            return_value=True,
        ):
            response = test_client.post("/sync")
            data = response.json()
            assert data["success"] is True

    def test_sync_returns_success_false_on_failure(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /sync returns success false when API call fails.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        with patch(
            "verification_ocr.api.server.sync_war_from_api",
            return_value=False,
        ):
            response = test_client.post("/sync")
            data = response.json()
            assert data["success"] is False

    def test_sync_returns_war_state(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that /sync returns current war state with calculated time.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        import time

        from verification_ocr.services import get_war_state

        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        # Set start time to 2.5 hours ago
        state.war_number = 132
        state.start_time = int((time.time() - 2.5 * 60 * 60) * 1000)

        try:
            with patch(
                "verification_ocr.api.server.sync_war_from_api",
                return_value=True,
            ):
                response = test_client.post("/sync")
                data = response.json()
                assert data["war_number"] == 132
                assert data["start_time"] is not None
                assert data["war_day"] == 3
                assert data["war_hour"] == 12
                assert data["war_minute"] == 0
        finally:
            state.war_number = original_number
            state.start_time = original_time


class TestIndexEndpoint:
    """Tests for / endpoint."""

    def test_index_returns_response(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that index returns a response.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        response = test_client.get("/")
        assert response.status_code == 200

    def test_index_returns_json_when_no_index_html(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that index returns JSON fallback when index.html doesn't exist.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        with patch(
            "verification_ocr.api.server.os.path.exists",
            return_value=False,
        ):
            response = test_client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Verification OCR Service"
            assert data["docs"] == "/docs"


class TestVerifyEndpoint:
    """Tests for /verify endpoint."""

    def _create_test_image(self, width: int = 3840, height: int = 2160) -> bytes:
        """
        Create a test image of specified size.

        Args:
            width (int): Image width.
            height (int): Image height.

        Returns:
            bytes: PNG encoded image bytes.
        """
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    def test_verify_with_valid_images(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test verify endpoint with valid images.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
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

    def test_verify_missing_image1(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test verify endpoint with missing image1.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        image_bytes = self._create_test_image()
        response = test_client.post(
            "/verify",
            files={
                "image2": ("test2.png", io.BytesIO(image_bytes), "image/png"),
            },
        )

        assert response.status_code == 422

    def test_verify_missing_image2(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test verify endpoint with missing image2.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
        image_bytes = self._create_test_image()
        response = test_client.post(
            "/verify",
            files={
                "image1": ("test1.png", io.BytesIO(image_bytes), "image/png"),
            },
        )

        assert response.status_code == 422

    def test_verify_returns_error_when_no_name_found(
        self,
        test_client: TestClient,
    ) -> None:
        """
        Test that verify returns error when no name found.

        Args:
            test_client (TestClient): Test client fixture.

        Returns:
            None
        """
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
        """
        Test that app is a FastAPI instance.

        Returns:
            None
        """
        assert isinstance(app, FastAPI)

    def test_app_has_health_route(self) -> None:
        """
        Test that app has /health route.

        Returns:
            None
        """
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_app_has_verify_route(self) -> None:
        """
        Test that app has /verify route.

        Returns:
            None
        """
        routes = [route.path for route in app.routes]
        assert "/verify" in routes

    def test_app_has_index_route(self) -> None:
        """
        Test that app has / route.

        Returns:
            None
        """
        routes = [route.path for route in app.routes]
        assert "/" in routes

    def test_app_has_war_route(self) -> None:
        """
        Test that app has /war route.

        Returns:
            None
        """
        routes = [route.path for route in app.routes]
        assert "/war" in routes

    def test_app_has_sync_route(self) -> None:
        """
        Test that app has /sync route.

        Returns:
            None
        """
        routes = [route.path for route in app.routes]
        assert "/sync" in routes
