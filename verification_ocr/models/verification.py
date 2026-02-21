"""Verification models."""

from pydantic import BaseModel, ConfigDict, Field

from verification_ocr.enums import Faction


class Verification(BaseModel):
    """User verification data extracted from game screenshots."""

    name: str | None = Field(default=None, description="Player name")
    level: int | None = Field(default=None, description="Player level")
    regiment: str | None = Field(
        default=None, description="Regiment name if player is in a regiment, None otherwise"
    )
    faction: Faction | None = Field(
        default=None, description="Player faction: 'colonial' or 'wardens'"
    )
    shard: str | None = Field(default=None, description="Game shard identifier")
    ingame_time: str | None = Field(default=None, description="In-game timestamp (day, HH:MM)")
    war_number: int | None = Field(default=None, description="Current war number")
    current_ingame_time: str | None = Field(
        default=None, description="Current in-game timestamp (day, HH:MM)"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "PlayerOne",
                "level": 25,
                "regiment": "[7-HP#123] 7th Hispanic Platoon",
                "faction": "colonial",
                "shard": "ABLE",
                "ingame_time": "267, 21:45",
                "war_number": 132,
                "current_ingame_time": "268, 14:30",
            }
        },
    )
