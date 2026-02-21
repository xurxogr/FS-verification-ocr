"""Health response model."""

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Health status")
    version: str = Field(description="Application version")
    tesseract_version: str | None = Field(
        default=None, description="Tesseract version if available"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "tesseract_version": "tesseract 5.3.0",
            }
        },
    )
