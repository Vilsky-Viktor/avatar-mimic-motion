FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip build-essential gnupg2 ca-certificates \
    libgl1 libglib2.0-0 ffmpeg python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# --- Poetry ---
ENV POETRY_VERSION=1.8.3
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install poetry==$POETRY_VERSION

COPY pyproject.toml poetry.lock* ./
COPY preprocess_image_job /app/preprocess_image_job

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

# RMBG-2.0 stack
RUN pip3 install --no-cache-dir huggingface_hub transformers accelerate safetensors

# Torch stack (CUDA 12.1 wheels)
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "-m", "preprocess_image_job.main"]