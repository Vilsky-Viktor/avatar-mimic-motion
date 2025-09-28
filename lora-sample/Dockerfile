# ===================================================================================
# Stage 1: Build dependencies using a smaller 'runtime' base image
# ===================================================================================
# USE: Smaller 'runtime' image. The 'devel' image is only needed for compiling.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# REMOVED: build-essential, cmake, git, etc. - Not needed if Python wheels are available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ PyTorch/cu118 trio that matches:
# torch 2.0.1  +  torchvision 0.15.2  +  torchaudio 2.0.2
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Core runtime deps for inference
# REMOVED: onnxruntime-gpu, bitsandbytes - These are for training or other model types.
RUN pip install --no-cache-dir --upgrade \
    google-cloud-storage==3.3.1 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    Pillow==10.4.0 \
    tqdm==4.66.4 \
    imageio==2.34.1 \
    imageio-ffmpeg==0.5.1 \
    safetensors==0.4.5 \
    diffusers==0.24.0 \
    transformers==4.32.1 \
    accelerate==0.27.2 \
    huggingface_hub==0.20.3

# Cache site-packages to copy into runtime
RUN mkdir -p /install && cp -r /usr/local/lib/python3.10/dist-packages /install/dist-packages

# ===============================
# Stage 2: Final runtime image
# ===============================
# USE: Smaller 'runtime' image for a smaller final build.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# REMOVED: libgl1, libglib2.0-0 - Not needed by opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# CUDA toolchain visibility (NVRTC, cuDNN, etc.)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Copy Python packages from builder stage
COPY --from=builder /install/dist-packages /usr/local/lib/python3.10/dist-packages

# Project code
COPY lora_sample /app/lora_sample

# Runtime env & QoL
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HUB_DISABLE_TELEMETRY=1
# REMOVED: INSIGHTFACE_ONNX_PROVIDERS - The insightface library is not used.

ENTRYPOINT ["python", "-m", "lora_sample.main"]