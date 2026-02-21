"""War service - fetches and manages war state from Foxhole API."""

import logging
from functools import lru_cache

import httpx

from verification_ocr.models import WarState

logger = logging.getLogger(__name__)

FOXHOLE_WAR_API_URL = "https://war-service-live.foxholeservices.com/api/worldconquest/war"


class WarService:
    """Service for managing war state."""

    def __init__(self) -> None:
        """Initialize war service with empty state."""
        self._state = WarState()

    @property
    def state(self) -> WarState:
        """
        Get the current war state.

        Returns:
            WarState: The current war state.
        """
        return self._state

    def initialize(
        self,
        war_number: int | None = None,
        start_time: int | None = None,
    ) -> None:
        """
        Initialize war state from settings.

        Args:
            war_number: War number from settings.
            start_time: Start time from settings.
        """
        self._state = WarState(war_number=war_number, start_time=start_time)

    async def fetch_from_api(self) -> WarState | None:
        """
        Fetch war data from Foxhole API.

        Returns:
            WarState | None: War state if successful, None otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(FOXHOLE_WAR_API_URL)
                response.raise_for_status()
                data = response.json()

                war_number = data.get("warNumber")
                start_time = data.get("conquestStartTime")

                if war_number is not None and start_time is not None:
                    logger.info(
                        f"Fetched war data from API: War {war_number}, start_time {start_time}"
                    )
                    return WarState(war_number=war_number, start_time=start_time)

                logger.warning("API response missing warNumber or conquestStartTime")
                return None

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching war data: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching war data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching war data: {e}")
            return None

    async def sync_from_api(self) -> bool:
        """
        Fetch war data from Foxhole API and update the state.

        Returns:
            bool: True if sync was successful, False otherwise.
        """
        result = await self.fetch_from_api()
        if result is not None:
            self._state = result
            return True
        return False


@lru_cache
def get_war_service() -> WarService:
    """
    Get the war service singleton.

    Returns:
        WarService: The war service instance.
    """
    return WarService()
