# syntax=docker/dockerfile:1

# Multi-stage build for smaller final image
# Build with: docker build --build-arg PYTHON_VERSION=3.13 .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml and __init__.py for dependency installation
COPY pyproject.toml README.md ./
COPY verification_ocr/__init__.py ./verification_ocr/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    # Strip debug symbols for smaller image
    find /opt/venv -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true

# Runtime stage
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install runtime dependencies (Tesseract OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY verification_ocr/ ./verification_ocr/
COPY static/ ./static/
COPY data/ ./data/
COPY pyproject.toml README.md ./

# Install package (no deps - already installed in venv)
RUN pip install --no-cache-dir --no-deps .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run server
CMD ["python", "-m", "uvicorn", "verification_ocr.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
