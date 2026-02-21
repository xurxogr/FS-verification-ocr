"""Region model for image processing."""

from pydantic import BaseModel, ConfigDict, Field


class Region(BaseModel):
    """A rectangular region defined by corner coordinates."""

    x1: int = Field(description="Left x coordinate")
    y1: int = Field(description="Top y coordinate")
    x2: int = Field(description="Right x coordinate")
    y2: int = Field(description="Bottom y coordinate")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "x1": 1390,
                "y1": 267,
                "x2": 1728,
                "y2": 332,
            }
        },
    )
