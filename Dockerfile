###########################
# 1) builder
###########################
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .

# CPU-only torch/torchvision come from this index
RUN pip install --prefix=/install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

###########################
# 2) runtime
###########################
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/main.py ./main.py
COPY app/model1_yolox/yolo11x-pose.pt ./model1_yolox/

EXPOSE 60000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60000","--workers", "2"]
