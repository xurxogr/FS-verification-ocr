"""War service - fetches and manages war state from Foxhole API."""

import logging

import httpx

logger = logging.getLogger(__name__)

FOXHOLE_WAR_API_URL = "https://war-service-live.foxholeservices.com/api/worldconquest/war"


class WarState:
    """Holds the current war state."""

    def __init__(self) -> None:
        """
        Initialize war state.

        Returns:
            None
        """
        self.war_number: int | None = None
        self.start_time: int | None = None

    def is_configured(self) -> bool:
        """
        Check if war state is configured.

        Returns:
            bool: True if both war_number and start_time are set.
        """
        return self.war_number is not None and self.start_time is not None


# Global war state instance
_war_state = WarState()


def get_war_state() -> WarState:
    """
    Get the global war state instance.

    Returns:
        WarState: The global war state.
    """
    return _war_state


async def sync_war_from_api() -> bool:
    """
    Fetch war data from Foxhole API and update the war state.

    Returns:
        bool: True if sync was successful, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(FOXHOLE_WAR_API_URL)
            response.raise_for_status()
            data = response.json()

            war_number = data.get("warNumber")
            start_time = data.get("conquestStartTime")

            if war_number is not None and start_time is not None:
                _war_state.war_number = war_number
                _war_state.start_time = start_time
                logger.info(f"Synced war data from API: War {war_number}, start_time {start_time}")
                return True
            else:
                logger.warning("API response missing warNumber or conquestStartTime")
                return False

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching war data: {e}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Request error fetching war data: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error fetching war data: {e}")
        return False


def initialize_war_state_from_settings(
    war_number: int | None,
    start_time: int | None,
) -> None:
    """
    Initialize war state from settings.

    Args:
        war_number (int | None): War number from settings.
        start_time (int | None): Start time from settings.

    Returns:
        None
    """
    if war_number is not None:
        _war_state.war_number = war_number
    if start_time is not None:
        _war_state.start_time = start_time
