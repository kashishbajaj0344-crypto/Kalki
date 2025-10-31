# Kalki v2.4 Production Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KALKI_ENV=production
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash kalki

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY modules/ ./modules/
COPY kalki_*.py ./
COPY main.py ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p data/vector_db data/logs data/sessions && \
    chown -R kalki:kalki /app

# Switch to non-root user
USER kalki

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python kalki_orchestrator.py --health-check || exit 1

# Expose port for web interface
EXPOSE 8000

# Default command
CMD ["python", "kalki_orchestrator.py", "--start-all"]

# Labels for container metadata
LABEL maintainer="Kalki Team"
LABEL version="2.4.0"
LABEL description="Kalki v2.4 - Advanced AI Orchestration Platform"