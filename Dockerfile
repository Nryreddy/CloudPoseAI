############################################
# 1) Builder stage: compile & install deps
############################################
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# only the bare minimum libs for opencv-python-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /install

# first install CPU-only PyTorch & torchvision
RUN pip install --prefix=/install \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

# then install the rest of your Python dependencies
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

############################################
# 2) Runtime stage: copy runtime bits only
############################################
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# again, only the libs needed at runtime for OpenCV-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# copy installed Python packages from the builder
COPY --from=builder /install /usr/local

WORKDIR /app

# copy only the files and dirs your server actually uses:
    COPY app/main.py       ./main.py
    COPY app/model1_yolox/  ./model1_yolox/

EXPOSE 60000

# match your 0.5-CPU / 512Mi limit by using one worker
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60000", "--workers", "1"]