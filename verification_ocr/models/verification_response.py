"""Verification response model."""

from pydantic import BaseModel, ConfigDict, Field

from verification_ocr.models.verification import Verification


class VerificationResponse(BaseModel):
    """Verification result response."""

    success: bool = Field(description="Whether verification was successful")
    error: str | None = Field(default=None, description="Error message if verification failed")
    verification: Verification | None = Field(
        default=None, description="Extracted verification data"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "success": True,
                "error": None,
                "verification": {
                    "name": "PlayerOne",
                    "level": 25,
                    "regiment": "[7HG] 7th Hispanic Platoon",
                    "colonial": True,
                    "shard": "ABLE",
                    "ingame_time": "267, 21:45",
                },
            }
        },
    )
