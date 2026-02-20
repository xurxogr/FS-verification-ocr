"""Response and request models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Health status")
    version: str = Field(description="Application version")
    tesseract_version: str | None = Field(
        default=None, description="Tesseract version if available"
    )


class VerificationRequest(BaseModel):
    """Verification request metadata."""

    pass


class VerificationResponse(BaseModel):
    """Verification result response."""

    success: bool = Field(description="Whether verification was successful")
    error: str | None = Field(default=None, description="Error message if verification failed")
    verification: dict | None = Field(
        default=None, description="Extracted verification data (name, level, regiment, etc.)"
    )


class WarResponse(BaseModel):
    """War information response."""

    war_number: int | None = Field(default=None, description="Current war number")
    war_day: int | None = Field(default=None, description="Current day of the war")
    war_hour: int | None = Field(default=None, description="Current in-game hour (0-23)")
    war_minute: int | None = Field(default=None, description="Current in-game minute (0-59)")
    start_time: int | None = Field(default=None, description="War start time in milliseconds")
