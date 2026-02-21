"""FastAPI application server."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from verification_ocr import __version__
from verification_ocr.api.dependencies import get_verification_service
from verification_ocr.core.settings import get_settings
from verification_ocr.core.utils import get_tesseract_version, setup_logging
from verification_ocr.models import HealthResponse, VerificationResponse, WarResponse
from verification_ocr.services import VerificationService, get_war_service

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")


def calculate_war_time(start_time: int) -> tuple[int, int, int]:
    """
    Calculate war day, hour, and minute from start time.

    Args:
        start_time (int): War start time in milliseconds.

        tuple[int, int, int]: (war_day, war_hour, war_minute)
    """
    current_time_ms = int(time.time() * 1000)
    elapsed_ms = current_time_ms - start_time
    elapsed_hours = elapsed_ms / (1000 * 60 * 60)

    # 1 real hour = 1 in-game day (game starts at Day 1, not Day 0)
    war_day = int(elapsed_hours) + 1

    # Convert fractional hour to in-game time (24h format)
    fraction = elapsed_hours % 1
    war_hour = int(fraction * 24)
    war_minute = int((fraction * 24 % 1) * 60)

    return war_day, war_hour, war_minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
    """
    settings = get_settings()

    # Setup logging
    setup_logging(settings=settings.logging)

    logger.info(f"Starting Verification OCR Service v{__version__}")

    # Check tesseract availability (hard requirement)
    tesseract_version = get_tesseract_version()
    if tesseract_version is None:
        raise RuntimeError(
            "Tesseract OCR is not installed or not accessible. "
            "Please install tesseract:\n"
            "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "  macOS: brew install tesseract\n"
            "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        )

    logger.info(f"Tesseract available: {tesseract_version}")

    # Initialize war state from settings
    war_service = get_war_service()
    war_service.initialize(
        war_number=settings.war.number,
        start_time=settings.war.start_time,
    )

    # If not configured, fetch from Foxhole API
    if not war_service.state.is_configured():
        logger.info("War settings not configured, fetching from Foxhole API...")
        await war_service.sync_from_api()

    if war_service.state.is_configured():
        war_day, war_hour, war_minute = calculate_war_time(start_time=war_service.state.start_time)
        logger.info(
            f"War {war_service.state.war_number} - Day {war_day}, {war_hour:02d}:{war_minute:02d}"
        )
    else:
        logger.warning("War state not configured and could not be fetched from API")

    yield

    logger.info("Shutting down Verification OCR Service")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

        FastAPI: The configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Verification OCR Service",
        description="Extract and compare information from image pairs using OCR",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_server.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for frontend
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()


@app.get("/", include_in_schema=False, response_model=None)
async def index() -> FileResponse | dict[str, str]:
    """
    Serve the frontend.

        FileResponse | dict[str, str]: The index.html file or a JSON message with docs link.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Verification OCR Service", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

        HealthResponse: Health status including version and tesseract availability.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        tesseract_version=get_tesseract_version(),
    )


@app.get("/war", response_model=WarResponse)
async def get_war_info() -> WarResponse:
    """
    Get current war information.

    Returns:
        WarResponse: War number and calculated war day/time based on start time.
    """
    war_service = get_war_service()
    state = war_service.state

    war_day = None
    war_hour = None
    war_minute = None

    if state.start_time is not None:
        war_day, war_hour, war_minute = calculate_war_time(start_time=state.start_time)

    return WarResponse(
        war_number=state.war_number,
        war_day=war_day,
        war_hour=war_hour,
        war_minute=war_minute,
        start_time=state.start_time,
    )


@app.post("/sync")
async def sync_war() -> dict[str, Any]:
    """
    Sync war data from Foxhole API and return updated war info.

    Returns:
        dict[str, Any]: Sync result with success status and current war state including time.
    """
    war_service = get_war_service()
    success = await war_service.sync_from_api()
    state = war_service.state

    war_day = None
    war_hour = None
    war_minute = None

    if state.start_time is not None:
        war_day, war_hour, war_minute = calculate_war_time(start_time=state.start_time)

    return {
        "success": success,
        "war_number": state.war_number,
        "war_day": war_day,
        "war_hour": war_hour,
        "war_minute": war_minute,
        "start_time": state.start_time,
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_images(
    image1: Annotated[UploadFile, File(description="First image")],
    image2: Annotated[UploadFile, File(description="Second image")],
    service: Annotated[VerificationService, Depends(get_verification_service)],
) -> VerificationResponse:
    """
    Verify user from two game screenshots.

    Upload two images to extract user information (name, level, regiment,
    faction, shard) using OCR and template matching.

    Args:
        image1 (UploadFile): First screenshot (profile or map).
        image2 (UploadFile): Second screenshot (profile or map).
        service (VerificationService): Injected verification service.

        VerificationResponse: Verification result with user info or error.
    """
    # Read image bytes
    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    # Process and compare
    return service.verify(
        image1_bytes=image1_bytes,
        image2_bytes=image2_bytes,
    )
