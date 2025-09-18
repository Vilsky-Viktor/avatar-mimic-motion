FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.11 python3.11-distutils python3.11-dev \
    build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to 3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Clone MimicMotion
RUN git clone https://github.com/Tencent/MimicMotion.git /app/MimicMotion

# Torch stack per environment.yaml (PyTorch 2.0.1 + CUDA 11.7)
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install \
      --force-reinstall torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
      --extra-index-url https://download.pytorch.org/whl/cu117

# Repo extras per environment.yaml
RUN python3.11 -m pip install \
    diffusers==0.27.0 transformers==4.32.1 \
    einops omegaconf \
    decord==0.6.0

# Our job deps
COPY pyproject.toml /app/
RUN python3.11 -m pip install poetry && \
    poetry config virtualenvs.create false && poetry install --no-root

# Job code
COPY mimic_motion_job /app/mimic_motion_job

# Defaults (override at runtime)
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV EXECUTION_ID=local-test

ENTRYPOINT ["python3.11", "-m", "mimic_motion_job.main"]