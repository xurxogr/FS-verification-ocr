"""War response model."""

from pydantic import BaseModel, ConfigDict, Field


class WarResponse(BaseModel):
    """War information response."""

    war_number: int | None = Field(default=None, description="Current war number")
    war_day: int | None = Field(default=None, description="Current day of the war")
    war_hour: int | None = Field(default=None, description="Current in-game hour (0-23)")
    war_minute: int | None = Field(default=None, description="Current in-game minute (0-59)")
    start_time: int | None = Field(default=None, description="War start time in milliseconds")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "war_number": 132,
                "war_day": 48,
                "war_hour": 12,
                "war_minute": 30,
                "start_time": 1770663602746,
            }
        },
    )
