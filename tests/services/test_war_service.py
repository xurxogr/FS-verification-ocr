"""Tests for war service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from verification_ocr.models import WarState
from verification_ocr.services import WarService, get_war_service


class TestWarState:
    """Tests for WarState model."""

    def test_initial_state(self) -> None:
        """Test initial war state values."""
        state = WarState()
        assert state.war_number is None
        assert state.start_time is None

    def test_is_configured_false_when_empty(self) -> None:
        """Test is_configured returns False when not configured."""
        state = WarState()
        assert state.is_configured() is False

    def test_is_configured_false_when_partial(self) -> None:
        """Test is_configured returns False when partially configured."""
        state = WarState(war_number=132)
        assert state.is_configured() is False

        state2 = WarState(start_time=1234567890)
        assert state2.is_configured() is False

    def test_is_configured_true_when_complete(self) -> None:
        """Test is_configured returns True when fully configured."""
        state = WarState(war_number=132, start_time=1234567890)
        assert state.is_configured() is True


class TestWarService:
    """Tests for WarService class."""

    def test_initial_state(self) -> None:
        """Test initial service state."""
        service = WarService()
        assert service.state.war_number is None
        assert service.state.start_time is None

    def test_initialize(self) -> None:
        """Test initialize sets state."""
        service = WarService()
        service.initialize(war_number=132, start_time=1234567890)
        assert service.state.war_number == 132
        assert service.state.start_time == 1234567890

    def test_initialize_partial(self) -> None:
        """Test initialize with partial values."""
        service = WarService()
        service.initialize(war_number=999)
        assert service.state.war_number == 999
        assert service.state.start_time is None

    @pytest.mark.asyncio
    async def test_fetch_from_api_success(self) -> None:
        """Test successful API fetch."""
        service = WarService()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "warNumber": 132,
            "conquestStartTime": 1770663602746,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await service.fetch_from_api()

            assert result is not None
            assert result.war_number == 132
            assert result.start_time == 1770663602746

    @pytest.mark.asyncio
    async def test_fetch_from_api_missing_data(self) -> None:
        """Test fetch returns None when data is missing."""
        service = WarService()

        mock_response = MagicMock()
        mock_response.json.return_value = {"warNumber": 132}
        mock_response.raise_for_status = MagicMock()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await service.fetch_from_api()
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_api_http_error(self) -> None:
        """Test fetch handles HTTP errors."""
        service = WarService()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    message="Not Found",
                    request=MagicMock(),
                    response=MagicMock(),
                )
            )

            result = await service.fetch_from_api()
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_api_request_error(self) -> None:
        """Test fetch handles request errors."""
        service = WarService()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError(message="Connection failed")
            )

            result = await service.fetch_from_api()
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_api_unexpected_error(self) -> None:
        """Test fetch handles unexpected errors."""
        service = WarService()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            result = await service.fetch_from_api()
            assert result is None

    @pytest.mark.asyncio
    async def test_sync_from_api_success(self) -> None:
        """Test successful sync updates state."""
        service = WarService()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "warNumber": 132,
            "conquestStartTime": 1770663602746,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await service.sync_from_api()

            assert result is True
            assert service.state.war_number == 132
            assert service.state.start_time == 1770663602746

    @pytest.mark.asyncio
    async def test_sync_from_api_failure(self) -> None:
        """Test failed sync returns False."""
        service = WarService()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError(message="Connection failed")
            )

            result = await service.sync_from_api()
            assert result is False


class TestGetWarService:
    """Tests for get_war_service function."""

    def test_returns_war_service(self) -> None:
        """Test get_war_service returns a WarService instance."""
        service = get_war_service()
        assert isinstance(service, WarService)

    def test_returns_same_instance(self) -> None:
        """Test get_war_service returns the same instance (singleton)."""
        service1 = get_war_service()
        service2 = get_war_service()
        assert service1 is service2
