"""War state model."""

from pydantic import BaseModel, ConfigDict, Field


class WarState(BaseModel):
    """Current war state."""

    war_number: int | None = Field(default=None, description="Current war number")
    start_time: int | None = Field(
        default=None, description="War start time in milliseconds (Unix timestamp)"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "war_number": 132,
                "start_time": 1770663602746,
            }
        },
    )

    def is_configured(self) -> bool:
        """
        Check if war state is configured.

        Returns:
            bool: True if both war_number and start_time are set.
        """
        return self.war_number is not None and self.start_time is not None
