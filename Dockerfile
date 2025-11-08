# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for PyMuPDF, Pillow, curl and text editor
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    nano \
    libmupdf-dev \
    mupdf-tools \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libtiff5-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .env from .env.example if it doesn't exist
RUN if [ ! -f .env ] && [ -f .env.example ]; then \
        cp .env.example .env; \
        echo "[Docker Build] Created .env from .env.example"; \
    fi

# Create necessary directories and clean up cache files
RUN mkdir -p \
    uploads \
    temp_uploads \
    app/flex_rag_data_location/documents \
    app/flex_rag_data_location/cache/summaries \
    logs \
    && find . -type f -name "*.pyc" -delete \
    && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Expose Streamlit default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Set the entrypoint
ENTRYPOINT ["streamlit", "run", "main.py"]

# Default Streamlit arguments
CMD ["--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
