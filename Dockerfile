# ---------- Stage 1: Build ----------
FROM python:3.10-slim as builder


# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apk add --no-cache \
build-base \
libjpeg-turbo-dev \
zlib-dev \
libffi-dev \
openssl-dev \
git

# Create and set working directory
WORKDIR /app

# Install pipenv dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
pip wheel --no-deps --wheel-dir=/wheels -r requirements.txt

# ---------- Stage 2: Final (distroless) ----------
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy your FastAPI app code
COPY . .

# Expose the port your app runs on
EXPOSE 60010

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "pose_api:app", "--host", "0.0.0.0", "--port", "60000"]
