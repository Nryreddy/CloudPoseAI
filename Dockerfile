############################################
# 1) Builder stage
############################################
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal runtime dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Install CPU-only dependencies first
COPY requirements.txt .
RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

############################################
# 2) Runtime stage
############################################
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Only copy necessary files
COPY app/main.py ./main.py
COPY app/model1_yolox/yolo11x-pose.pt ./model1_yolox/

EXPOSE 60000

# Run with single worker to match resource limits
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60000", "--workers", "1"]