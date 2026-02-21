# Verification OCR Service

A FastAPI service for extracting and comparing information from Foxhole game screenshots using OpenCV and Tesseract OCR.

## Quick Start

### Docker

```bash
docker compose up --build
```

Then open http://localhost:8000 for the web interface or http://localhost:8000/docs for the API documentation.

#### Python Version Selection

You can choose between Python 3.12 or 3.13:

```bash
# Use Python 3.13
docker compose build --build-arg PYTHON_VERSION=3.13

# Use Python 3.12 (default)
docker compose build --build-arg PYTHON_VERSION=3.12
```

### Local Development

```bash
# Install dependencies
pip install -e .

# Run server
vocr server
# or
python -m uvicorn verification_ocr.api.server:app --reload
```

## Configuration

Environment variables (prefix `VOCR_`). Can be set in a `.env` file.

### API Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCR_API_SERVER__HOST` | `127.0.0.1` | Server host |
| `VOCR_API_SERVER__PORT` | `8000` | Server port |
| `VOCR_API_SERVER__API_KEY` | `None` | API key for authentication (disabled if not set) |
| `VOCR_API_SERVER__SERVE_FRONTEND` | `true` | Serve the frontend web interface |
| `VOCR_API_SERVER__CORS_ALLOW_ORIGINS` | `[]` | CORS allowed origins (empty = no CORS) |
| `VOCR_API_SERVER__RATE_LIMIT` | `10/minute` | Rate limit for `/verify` endpoint |
| `VOCR_API_SERVER__MAX_UPLOAD_SIZE` | `52428800` | Max upload size in bytes (default 50MB) |

### OCR Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCR_OCR__LANGUAGE` | `eng+fra+deu+por+rus+chi_sim` | Tesseract language codes |
| `VOCR_OCR__DEBUG_MODE` | `false` | Save debug images with detected regions |

### War Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCR_WAR__NUMBER` | Auto-fetched | Current war number |
| `VOCR_WAR__START_TIME` | Auto-fetched | War start time (Unix ms) |

### Logging Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCR_LOGGING__LOG_LEVEL` | `INFO` | Log level |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Web interface (if enabled) |
| `/health` | GET | No | Health check |
| `/war` | GET | No | Current war information |
| `/verify` | POST | Yes* | Upload two images for verification |
| `/sync` | POST | Yes* | Sync war data from Foxhole API |
| `/docs` | GET | No | OpenAPI documentation |

*Authentication required only if `VOCR_API_SERVER__API_KEY` is configured.

## Authentication

When `VOCR_API_SERVER__API_KEY` is set, the `/verify` and `/sync` endpoints require the `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/sync \
  -H "X-API-Key: your-secret-key"
```

## Security Features

- **Rate limiting**: Configurable rate limit on `/verify` endpoint
- **Upload size limits**: Maximum file size validation (default 50MB)
- **Security headers**: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy
- **HSTS**: Strict-Transport-Security header when behind HTTPS proxy
- **Path traversal protection**: Validated file paths for debug output
- **API key authentication**: Optional authentication for sensitive endpoints
