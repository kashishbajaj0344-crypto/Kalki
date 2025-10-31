# Simulation Sandbox Dockerfile
# Provides a secure, isolated environment for running simulations

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create simulation user with limited privileges
RUN useradd --create-home --shell /bin/bash simuser && \
    mkdir -p /app && \
    chown simuser:simuser /app

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER simuser

# Set environment variables for security
ENV PYTHONPATH=/app
ENV SANDBOX_MODE=true
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "/app/simulation.py"]