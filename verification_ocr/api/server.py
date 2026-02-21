"""FastAPI application server."""

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from verification_ocr import __version__
from verification_ocr.api.dependencies import get_verification_service, verify_api_key
from verification_ocr.core.settings import get_settings
from verification_ocr.core.utils import calculate_war_time, get_tesseract_version, setup_logging
from verification_ocr.models import HealthResponse, VerificationResponse, WarResponse
from verification_ocr.services import VerificationService, get_war_service

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # Only add HSTS if behind HTTPS proxy (check X-Forwarded-Proto)
        if request.headers.get("X-Forwarded-Proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Handle rate limit exceeded errors."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
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

    if war_service.state.is_configured() and war_service.state.start_time is not None:
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

    # Add rate limiter state and exception handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS middleware - only add if origins are specified
    if settings.api_server.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api_server.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # Mount static files for frontend (if enabled)
    if settings.api_server.serve_frontend and os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()


@app.get("/", include_in_schema=False, response_model=None)
async def index() -> FileResponse | dict[str, str]:
    """
    Serve the frontend.

        FileResponse | dict[str, str]: The index.html file or a JSON message with docs link.
    """
    settings = get_settings()

    # Return JSON if frontend is disabled
    if not settings.api_server.serve_frontend:
        return {"message": "Verification OCR Service", "docs": "/docs"}

    index_path = os.path.realpath(os.path.join(STATIC_DIR, "index.html"))
    # Ensure path is within STATIC_DIR (prevent path traversal)
    if not index_path.startswith(os.path.realpath(STATIC_DIR)):
        return {"message": "Verification OCR Service", "docs": "/docs"}
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
async def sync_war(
    _api_key: Annotated[str | None, Depends(verify_api_key)],
) -> dict[str, Any]:
    """
    Sync war data from Foxhole API and return updated war info.

    Requires API key authentication if configured.

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
@limiter.limit(lambda: get_settings().api_server.rate_limit)
async def verify_images(
    request: Request,
    image1: Annotated[UploadFile, File(description="First image")],
    image2: Annotated[UploadFile, File(description="Second image")],
    service: Annotated[VerificationService, Depends(get_verification_service)],
    _api_key: Annotated[str | None, Depends(verify_api_key)],
) -> VerificationResponse:
    """
    Verify user from two game screenshots.

    Upload two images to extract user information (name, level, regiment,
    faction, shard) using OCR and template matching.

    Requires API key authentication if configured.

    Args:
        request (Request): The request object (required for rate limiting).
        image1 (UploadFile): First screenshot (profile or map).
        image2 (UploadFile): Second screenshot (profile or map).
        service (VerificationService): Injected verification service.

        VerificationResponse: Verification result with user info or error.
    """
    settings = get_settings()
    max_size = settings.api_server.max_upload_size

    # Read and validate image sizes
    image1_bytes = await image1.read()
    if len(image1_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image 1 exceeds maximum size of {max_size // (1024 * 1024)}MB",
        )

    image2_bytes = await image2.read()
    if len(image2_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image 2 exceeds maximum size of {max_size // (1024 * 1024)}MB",
        )

    # Process and compare
    return service.verify(
        image1_bytes=image1_bytes,
        image2_bytes=image2_bytes,
    )
