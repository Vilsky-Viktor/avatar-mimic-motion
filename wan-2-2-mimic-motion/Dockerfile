# ==========================
# Wan2.2 — CUDA 12.4, PyTorch 2.4 (Optimized)
# ==========================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1 \
    # FIX: Corrected typo '0' to '9.0' for Hopper support
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    USE_NINJA=1 \
    # FIX: Reduced parallel jobs to prevent memory exhaustion during build
    MAX_JOBS=4

# Build & runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git git-lfs \
    build-essential ninja-build cmake pkg-config \
    ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# ---------- Python 3.12 ----------
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Virtualenv
ENV VENV=/opt/venv
RUN python3.12 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Base Python tooling
RUN pip install --upgrade pip setuptools wheel packaging

# ---------- Clone Wan2.2 ----------
ARG WAN_REPO=https://github.com/Wan-Video/Wan2.2
ARG WAN_DIR=/app/Wan2.2
RUN git clone --depth=1 "$WAN_REPO" "$WAN_DIR"
WORKDIR $WAN_DIR
# If the repo uses LFS files, fetch them (won't hurt if none exist)
RUN git lfs pull || true

# ---------- Install All Dependencies in a Single Layer ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.0 torchvision==0.20.0 \
 && pip install -r requirements.txt \
 && pip install -r requirements_animate.txt \
 && pip install -r requirements_s2v.txt \
 && pip install google-cloud-storage==2.18.0 \
    huggingface-hub==0.30 \
    moviepy \
 && pip install --upgrade diffusers \
 && pip uninstall -y onnxruntime onnxruntime-silicon \
 && pip install onnxruntime-gpu==1.19.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# ---------- App code ----------
WORKDIR /app
COPY mimic_motion /app/mimic_motion

CMD ["python", "-m", "mimic_motion.main"]

