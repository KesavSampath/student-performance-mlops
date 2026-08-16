# ── Dockerfile ───────────────────────────────────────────────────────────────
# FastAPI backend for Student Performance Prediction
# Base: Python 3.11-slim (no GPU, CPU-only)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies (for scikit-learn, shap, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run FastAPI via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
