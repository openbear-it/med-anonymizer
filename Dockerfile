# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed by some transformers deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only (much smaller; works on AMD64 and ARM64)
# The CPU index covers AMD64; for ARM64 pip falls back to the standard PyPI CPU wheel.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install --no-cache-dir torch && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: model download ───────────────────────────────────────────────────
FROM python:3.11-slim AS model-downloader

WORKDIR /models

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

ENV HF_HOME=/models/hf_cache
ENV TRANSFORMERS_CACHE=/models/hf_cache

# Pre-download the default model (baked into the image → no cold-start download)
RUN python - <<'EOF'
from transformers import pipeline
import logging
logging.basicConfig(level=logging.INFO)
print("Downloading OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1 …")
pipe = pipeline("ner", model="OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1",
                aggregation_strategy="simple", device=-1)
print("Model ready.")
EOF


# ── Stage 3: final runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Med-Anonymizer" \
      org.opencontainers.image.description="Clinical PII de-identification REST API powered by OpenMed" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://huggingface.co/OpenMed"

WORKDIR /app

# Non-root user for security
RUN groupadd -r medanon && useradd -r -g medanon medanon

# Copy packages and model cache from previous stages
COPY --from=builder      /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder      /usr/local/bin             /usr/local/bin
COPY --from=model-downloader /models/hf_cache       /app/hf_cache

# Application code
COPY app/ ./app/

# Point HuggingFace to the baked-in cache
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN chown -R medanon:medanon /app

USER medanon

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
