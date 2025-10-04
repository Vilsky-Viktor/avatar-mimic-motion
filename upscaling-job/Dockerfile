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

# Torch stack (CUDA 11.7)
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      --force-reinstall torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
      --extra-index-url https://download.pytorch.org/whl/cu117

# Real-ESRGAN & friends
RUN python -m pip install \
    realesrgan==0.3.0 \
    basicsr==1.4.2 \
    facexlib==0.3.0 \
    gfpgan==1.3.8 \
    opencv-python==4.11.0.86 \
    numpy==1.26.4 \
    imageio==2.34.1 imageio-ffmpeg==0.4.9 \
    tqdm==4.66.4 \
    google-cloud-storage==2.18.0

# App
COPY upscaling_job /app/upscaling_job

ENTRYPOINT ["python", "-m", "upscaling_job.main"]