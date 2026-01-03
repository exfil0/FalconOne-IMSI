# FalconOne Intelligence Platform - Docker Image
# Version 1.8.0 - RANSacked Integration Complete
# Multi-stage build for optimal image size and security
# Includes: RANSacked vulnerability auditor, XSS protection, rate limiting, caching

# ==================== Stage 1: Base Dependencies ====================
FROM ubuntu:22.04 as base

LABEL maintainer="FalconOne Intelligence"
LABEL version="1.9.0"
LABEL description="Multi-generation IMSI/TMSI catcher with AI/ML, SDR, and RANSacked vulnerability auditing"
LABEL org.opencontainers.image.source="https://github.com/falconone/platform"
LABEL org.opencontainers.image.documentation="https://docs.falconone.io"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# RANSacked environment variables (Phase 5-7 optimizations)
ENV RANSACKED_CACHE_SIZE=128
ENV RANSACKED_RATE_LIMIT_SCAN=10
ENV RANSACKED_RATE_LIMIT_PACKET=20
ENV RANSACKED_RATE_LIMIT_STATS=60

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libuhd-dev \
    uhd-host \
    gnuradio \
    gr-gsm \
    tshark \
    wireshark-common \
    soapysdr-tools \
    libsoapysdr-dev \
    soapysdr-module-all \
    swig \
    libyaml-dev \
    libfftw3-dev \
    libmbedtls-dev \
    libconfig++-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ==================== Stage 2: Python Dependencies ====================
FROM base as python-deps

COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# ==================== Stage 3: Application Build ====================
FROM python-deps as app-build

# Copy application code
COPY falconone/ /app/falconone/
COPY config/ /app/config/
COPY run.py /app/
COPY deploy.sh /app/

# Create necessary directories (including logs volume for RANSacked audit trails)
RUN mkdir -p /app/logs /app/logs/audit /app/data /app/captures /app/models /app/checkpoints

# Set permissions
RUN chmod +x /app/run.py /app/deploy.sh

# ==================== Stage 4: Production Image ====================
FROM app-build as production

# Non-root user for security
RUN useradd -m -u 1000 falconone && \
    chown -R falconone:falconone /app

# Define volumes for persistent data and logs (RANSacked audit logs)
VOLUME ["/app/logs", "/app/data", "/app/captures"]

USER falconone

# Expose ports
# 5000: HTTP API
# 8080: Monitoring dashboard
# 38412: NGAP (5G)
# 36412: S1AP (LTE)
EXPOSE 5000 8080 38412 36412

# Health check - tests RANSacked statistics endpoint availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/audit/ransacked/stats', timeout=5)" || exit 1

# Default command
CMD ["python3", "/app/run.py"]

# ==================== Stage 5: Development Image ====================
FROM app-build as development

# Install development tools
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Keep root for development
USER root

CMD ["/bin/bash"]
