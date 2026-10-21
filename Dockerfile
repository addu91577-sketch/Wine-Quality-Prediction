# Production Multi-Stage Dockerfile for Wine Quality Prediction Service
# Serves high-throughput FastAPI REST API and optional Streamlit Dashboard

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for scikit-learn and network healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and data
COPY src/ /app/src/
COPY models/ /app/models/
COPY Data/ /app/Data/
COPY conftest.py pytest.ini /app/
COPY tests/ /app/tests/

# Create non-root system user for security compliance
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI default port and Streamlit port
EXPOSE 8000 8501

# Healthcheck probe querying FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution: Run FastAPI with Uvicorn production server
CMD ["uvicorn", "src.production_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
