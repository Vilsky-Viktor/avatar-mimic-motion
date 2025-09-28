FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.11 python3.11-distutils python3.11-dev \
    build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Make python point to 3.11 ---
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# --- pip ---
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# --- Poetry ---
RUN python3.11 -m pip install poetry

# --- Copy project ---
COPY pyproject.toml poetry.lock* /app/
COPY avatar_pose_job /app/avatar_pose_job

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118

RUN python3.11 -m pip install \
    onnxruntime==1.15.0 \
    onnxruntime-gpu==1.15.1

RUN python3.11 -m poetry config virtualenvs.create false && \
    python3.11 -m poetry install --no-root --no-interaction --no-ansi

# --- Defaults ---
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV AI_AVATAR_ID=""

ENTRYPOINT ["python3.11", "-m", "avatar_pose_job.main"]