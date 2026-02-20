# Verification OCR Service

A FastAPI service for extracting and comparing information from image pairs using OpenCV and Tesseract OCR.

## Quick Start

### Docker

```bash
docker compose up --build
```

Then open http://localhost:8000 for the web interface or http://localhost:8000/docs for the API documentation.

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

Environment variables (prefix `VOCR_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VOCR_API_SERVER__HOST` | `0.0.0.0` | Server host |
| `VOCR_API_SERVER__PORT` | `8000` | Server port |
| `VOCR_OCR__LANGUAGE` | `eng` | Tesseract language |
| `VOCR_LOGGING__LEVEL` | `INFO` | Log level |

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /verify` - Upload two images for verification
- `GET /docs` - OpenAPI documentation
