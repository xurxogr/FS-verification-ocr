"""Verification models."""

from pydantic import BaseModel, Field


class Verification(BaseModel):
    """User verification data extracted from game screenshots."""

    name: str | None = Field(default=None, description="Player name")
    level: int | None = Field(default=None, description="Player level")
    regiment: str | None = Field(
        default=None, description="Regiment name if player is in a regiment, None otherwise"
    )
    colonial: bool | None = Field(default=None, description="Whether player is colonial faction")
    shard: str | None = Field(default=None, description="Game shard identifier")
    ingame_time: str | None = Field(default=None, description="In-game timestamp (day, HH:MM)")
