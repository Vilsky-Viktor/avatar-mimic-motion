FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.11 python3.11-distutils python3.11-dev \
    build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    libjpeg-dev libpng-dev \
    cuda-nvrtc-11-8 cuda-nvrtc-dev-11-8 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip

# Torch stack for CUDA 11.8
RUN python3.11 -m pip install \
  --force-reinstall torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Libs
RUN python3.11 -m pip install \
  diffusers==0.27.2 \
  transformers==4.43.3 \
  accelerate==0.34.2 \
  peft==0.11.1 \
  huggingface-hub==0.23.5 \
  safetensors==0.4.4 \
  einops omegaconf \
  decord==0.6.0 \
  google-cloud-storage==2.18.0 \
  PyYAML==6.0.2 \
  opencv-python==4.11.0.86 \
  numpy==1.26.4 \
  matplotlib==3.10.0 \
  onnxruntime-gpu==1.17.1 \
  av==10.0.0

# Make sure CUDA libs are visible (usually already set, but harmless)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY mimic_motion_job /app/mimic_motion_job
COPY MimicMotion /app/MimicMotion

ENV JOBS_BUCKET=billion-ai-girls-asia
ENV EXECUTION_ID=local-test

ENTRYPOINT ["python3.11", "-m", "mimic_motion_job.main"]