"""Image regions model for image processing."""

from pydantic import BaseModel, ConfigDict, Field

from verification_ocr.models.region import Region


class ImageRegions(BaseModel):
    """Collection of detected regions in an image."""

    username: Region = Field(description="Username text region")
    icon: Region = Field(description="Faction icon region")
    level: Region = Field(description="Level text region")
    regiment: Region = Field(description="Regiment name region")
    shard: Region = Field(description="Shard/time region")
    scale_factor: float = Field(description="Image scale factor relative to 4K")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "username": {"x1": 1390, "y1": 267, "x2": 1728, "y2": 332},
                "icon": {"x1": 1797, "y1": 267, "x2": 1912, "y2": 332},
                "level": {"x1": 1927, "y1": 267, "x2": 2173, "y2": 332},
                "regiment": {"x1": 1858, "y1": 527, "x2": 2688, "y2": 578},
                "shard": {"x1": 64, "y1": 1968, "x2": 358, "y2": 2032},
                "scale_factor": 1.0,
            }
        },
    )
