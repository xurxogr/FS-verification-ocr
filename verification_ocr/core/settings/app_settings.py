"""Application settings using pydantic-settings."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIServerSettings(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    cors_allow_origins: list[str] = Field(default=["*"], description="CORS allowed origins")


class VerificationSettings(BaseModel):
    """Verification configuration."""

    max_ingame_time_diff: int = Field(
        default=25,
        description="Maximum allowed in-game time difference in hours",
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
    scale_factor: int = Field(default=4, description="Scale factor for small text extraction")
    clahe_clip_limit: float = Field(
        default=2.0, description="CLAHE clip limit for contrast enhancement"
    )
    clahe_grid_size: int = Field(default=8, description="CLAHE grid size for contrast enhancement")
    # Base dimensions for 4K (2160p) reference resolution
    base_height: int = Field(
        default=2160, description="Base height for scaling calculations (4K reference)"
    )
    base_box_width: int = Field(default=84, description="Base box width at 4K resolution")
    base_box_height: int = Field(default=64, description="Base box height at 4K resolution")
    debug_mode: bool = Field(default=False, description="Save debug images with detected regions")
    debug_output_dir: str = Field(default="screenshots", description="Directory for debug images")


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
