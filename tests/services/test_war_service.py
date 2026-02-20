"""Tests for war service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from verification_ocr.services.war_service import (
    WarState,
    get_war_state,
    initialize_war_state_from_settings,
    sync_war_from_api,
)


class TestWarState:
    """Tests for WarState class."""

    def test_initial_state(self) -> None:
        """
        Test initial war state values.

        Returns:
            None
        """
        state = WarState()
        assert state.war_number is None
        assert state.start_time is None

    def test_is_configured_false_when_empty(self) -> None:
        """
        Test is_configured returns False when not configured.

        Returns:
            None
        """
        state = WarState()
        assert state.is_configured() is False

    def test_is_configured_false_when_partial(self) -> None:
        """
        Test is_configured returns False when partially configured.

        Returns:
            None
        """
        state = WarState()
        state.war_number = 132
        assert state.is_configured() is False

        state2 = WarState()
        state2.start_time = 1234567890
        assert state2.is_configured() is False

    def test_is_configured_true_when_complete(self) -> None:
        """
        Test is_configured returns True when fully configured.

        Returns:
            None
        """
        state = WarState()
        state.war_number = 132
        state.start_time = 1234567890
        assert state.is_configured() is True


class TestGetWarState:
    """Tests for get_war_state function."""

    def test_returns_war_state(self) -> None:
        """
        Test get_war_state returns a WarState instance.

        Returns:
            None
        """
        state = get_war_state()
        assert isinstance(state, WarState)

    def test_returns_same_instance(self) -> None:
        """
        Test get_war_state returns the same instance.

        Returns:
            None
        """
        state1 = get_war_state()
        state2 = get_war_state()
        assert state1 is state2


class TestInitializeWarStateFromSettings:
    """Tests for initialize_war_state_from_settings function."""

    def test_sets_war_number(self) -> None:
        """
        Test that war_number is set from settings.

        Returns:
            None
        """
        state = get_war_state()
        original_number = state.war_number

        initialize_war_state_from_settings(war_number=999, start_time=None)
        assert state.war_number == 999

        # Restore
        state.war_number = original_number

    def test_sets_start_time(self) -> None:
        """
        Test that start_time is set from settings.

        Returns:
            None
        """
        state = get_war_state()
        original_time = state.start_time

        initialize_war_state_from_settings(war_number=None, start_time=9999999)
        assert state.start_time == 9999999

        # Restore
        state.start_time = original_time

    def test_sets_both(self) -> None:
        """
        Test that both values are set from settings.

        Returns:
            None
        """
        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

        initialize_war_state_from_settings(war_number=888, start_time=8888888)
        assert state.war_number == 888
        assert state.start_time == 8888888

        # Restore
        state.war_number = original_number
        state.start_time = original_time


class TestSyncWarFromApi:
    """Tests for sync_war_from_api function."""

    @pytest.mark.asyncio
    async def test_successful_sync(self) -> None:
        """
        Test successful API sync.

        Returns:
            None
        """
        state = get_war_state()
        original_number = state.war_number
        original_time = state.start_time

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

            result = await sync_war_from_api()

            assert result is True
            assert state.war_number == 132
            assert state.start_time == 1770663602746

        # Restore
        state.war_number = original_number
        state.start_time = original_time

    @pytest.mark.asyncio
    async def test_missing_war_number(self) -> None:
        """
        Test sync fails when warNumber is missing.

        Returns:
            None
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "conquestStartTime": 1770663602746,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await sync_war_from_api()
            assert result is False

    @pytest.mark.asyncio
    async def test_missing_start_time(self) -> None:
        """
        Test sync fails when conquestStartTime is missing.

        Returns:
            None
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "warNumber": 132,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await sync_war_from_api()
            assert result is False

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        """
        Test sync handles HTTP errors.

        Returns:
            None
        """
        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    message="Not Found",
                    request=MagicMock(),
                    response=MagicMock(),
                )
            )

            result = await sync_war_from_api()
            assert result is False

    @pytest.mark.asyncio
    async def test_request_error(self) -> None:
        """
        Test sync handles request errors.

        Returns:
            None
        """
        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError(message="Connection failed")
            )

            result = await sync_war_from_api()
            assert result is False

    @pytest.mark.asyncio
    async def test_unexpected_error(self) -> None:
        """
        Test sync handles unexpected errors.

        Returns:
            None
        """
        with patch("verification_ocr.services.war_service.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            result = await sync_war_from_api()
            assert result is False
