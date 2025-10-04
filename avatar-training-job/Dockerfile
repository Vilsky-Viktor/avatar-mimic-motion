FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# System deps + NVRTC (needed by cuDNN/PyTorch dyn kernels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.11 python3.11-distutils python3.11-dev \
    build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    libjpeg-dev libpng-dev \
    cuda-nvrtc-11-8 cuda-nvrtc-dev-11-8 \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# PyTorch cu118 (same major as your working job)
RUN python3.11 -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118

# Core libs (keep your diffs pins; ORT GPU 1.17.1 as in your working job)
RUN python3.11 -m pip install \
  diffusers==0.27.0 \
  transformers==4.43.3 \
  accelerate==0.30.1 \
  huggingface-hub==0.23.5 \
  safetensors==0.4.4 \
  einops omegaconf \
  google-cloud-storage==2.18.0 \
  opencv-python-headless==4.10.0.84 \
  pillow==10.4.0 \
  imageio==2.34.1 \
  tqdm==4.66.4 \
  numpy==1.26.4 \
  insightface==0.7.3 \
  onnxruntime-gpu==1.17.1

# MimicMotion source (import path for UNet)
RUN git clone --depth=1 https://github.com/Tencent/MimicMotion.git /opt/MimicMotion
ENV PYTHONPATH=/app:/opt/MimicMotion:$PYTHONPATH

# Your trainer code
COPY avatar_training_job /app/avatar_training_job

ENTRYPOINT ["python3.11", "-m", "avatar_training_job.main"]