"""Application settings using pydantic-settings."""

import os

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIServerSettings(BaseModel):
    """API server configuration."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of uvicorn workers")
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="CORS allowed origins (empty = no CORS)"
    )
    max_upload_size: int = Field(
        default=50 * 1024 * 1024, description="Max upload size in bytes (default 50MB)"
    )
    rate_limit: str = Field(default="10/minute", description="Rate limit for verification endpoint")
    api_key: str | None = Field(
        default=None, description="API key for authentication (None = auth disabled)"
    )
    serve_frontend: bool = Field(
        default=True, description="Serve the frontend static files and index page"
    )


class VerificationSettings(BaseModel):
    """Verification configuration."""

    max_ingame_time_diff: int = Field(
        default=25,
        description="Maximum allowed in-game time difference in days",
    )


class OCRSettings(BaseModel):
    """OCR processing configuration."""

    tesseract_cmd: str | None = Field(
        default=None, description="Path to tesseract binary (None = auto-detect)"
    )
    language: str = Field(
        default="eng+fra+deu+por+rus+chi_sim",
        description="Tesseract language codes (+ separated)",
    )
    colonial_icon_path: str | None = Field(
        default="data/colonial_icon.png",
        description="Path to colonial faction icon for template matching",
    )
    scale_factor: int = Field(
        default=4, ge=1, le=10, description="Scale factor for small text extraction"
    )
    clahe_clip_limit: float = Field(
        default=2.0, ge=0.1, le=10.0, description="CLAHE clip limit for contrast enhancement"
    )
    clahe_grid_size: int = Field(
        default=8, ge=2, le=32, description="CLAHE grid size for contrast enhancement"
    )
    # Base dimensions for 4K (2160p) reference resolution
    base_height: int = Field(
        default=2160, ge=480, description="Base height for scaling calculations (4K reference)"
    )
    base_box_width: int = Field(default=84, ge=1, description="Base box width at 4K resolution")
    base_box_height: int = Field(default=64, ge=1, description="Base box height at 4K resolution")
    debug_mode: bool = Field(default=False, description="Save debug images with detected regions")
    debug_output_dir: str = Field(default="screenshots", description="Directory for debug images")

    @field_validator("tesseract_cmd")
    @classmethod
    def validate_tesseract_cmd(cls, v: str | None) -> str | None:
        """Validate tesseract command path exists and is executable."""
        if v is None:
            return v
        if not os.path.isfile(v):
            raise ValueError(f"Tesseract binary not found: {v}")
        if not os.access(v, os.X_OK):
            raise ValueError(f"Tesseract binary not executable: {v}")
        return v

    @field_validator("colonial_icon_path")
    @classmethod
    def validate_icon_path(cls, v: str | None) -> str | None:
        """Validate colonial icon path exists."""
        if v is None:
            return v
        if not os.path.isfile(v):
            # Don't raise, just warn - icon is optional
            return None
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language string is not empty."""
        if not v or not v.strip():
            raise ValueError("Language string cannot be empty")
        return v.strip()


class WarSettings(BaseModel):
    """War configuration."""

    number: int | None = Field(default=None, description="Current war number")
    start_time: int | None = Field(
        default=None, description="War start time in milliseconds (Unix timestamp)"
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    loggers: dict[str, str] = Field(default={}, description="Loggers and their levels")
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        description="Log format",
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Log date format")
    rotate_logs: bool = Field(default=False, description="Rotate logs daily")
    log_file: str | None = Field(default=None, description="Log file to write to")


class AppSettings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="VOCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_server: APIServerSettings = Field(default_factory=APIServerSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    war: WarSettings = Field(default_factory=WarSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
