# ============================================================
# PRODUCTION DOCKERFILE
# RECEIPT OCR & DOCUMENT INTELLIGENT EXTRACTION
# ============================================================

FROM python:3.11-slim

# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        curl \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff6 \
        libwebp7 \
        libopenjp2-7 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install \
        --no-cache-dir \
        -r requirements.txt

# ============================================================
# APPLICATION SOURCE
# ============================================================

COPY . .

# ============================================================
# INSTALL PROJECT
# ============================================================

RUN python -m pip install \
        --no-cache-dir \
        .

# ============================================================
# RUNTIME DIRECTORIES
# ============================================================

RUN mkdir -p \
        /app/artifacts \
        /app/artifacts/data_ingestion \
        /app/artifacts/data_validation \
        /app/artifacts/image_preprocessing \
        /app/artifacts/ocr_engine \
        /app/artifacts/ocr_text_cleaning \
        /app/artifacts/field_extraction_engine \
        /app/artifacts/confidence_and_reliability_engine \
        /app/artifacts/json_generation \
        /app/artifacts/financial_summary \
        /app/artifacts/evaluation \
        /app/artifacts/inference \
        /app/artifacts/mlflow \
        /app/data \
        /app/logs

# ============================================================
# NON-ROOT USER
# ============================================================

RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        --uid 1000 \
        appuser && \
    chown -R appuser:appuser /app

USER appuser

# ============================================================
# NETWORK
# ============================================================

EXPOSE 8000

# ============================================================
# HEALTHCHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl \
        --fail \
        http://127.0.0.1:8000/health/live || exit 1

# ============================================================
# APPLICATION
# ============================================================

CMD [ "uvicorn", "app:app","--host","0.0.0.0","--port","8000","--proxy-headers"]