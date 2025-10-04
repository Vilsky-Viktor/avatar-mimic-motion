# CUDA 12.4.1 + cuDNN runtime on Ubuntu 22.04 (valid tag)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/.cache \
    FACEFUSION_HOME=/opt/facefusion

# --- OS + Python ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl git ffmpeg \
      libglib2.0-0 libgl1 \
      build-essential pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# --- Base Python libs (no UI) ---
# Pin NumPy < 2.3.0 so OpenCV wheels resolve; plus common libs FaceFusion uses in headless mode
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir \
      "numpy<2.3.0,>=2.0.0" \
      google-cloud-storage==2.18.2 \
      tenacity==8.2.3 \
      pillow>=10.3.0 \
      tqdm \
      pyyaml \
      scipy \
      requests

# --- FaceFusion source ---
RUN git clone --depth=1 https://github.com/facefusion/facefusion ${FACEFUSION_HOME}

# --- Headless runtime deps only (skip FaceFusion's UI requirements.txt) ---
# Use a published ORT GPU wheel (CUDA 12.x), ONNX, headless OpenCV, InsightFace (will compile C++ ext -> needs build-essential)
RUN python3.11 -m pip install --no-cache-dir \
      onnxruntime-gpu==1.22.0 \
      onnx==1.19.0 \
      opencv-python-headless==4.12.0.88 \
      insightface==0.7.3

# Cache dir for model downloads (writable by Vertex runtime user)
RUN mkdir -p ${XDG_CACHE_HOME} && chmod -R 777 ${XDG_CACHE_HOME}

# (Optional) Quick probe
RUN python3.11 -c "import onnxruntime as ort, numpy as np, cv2; print('ORT providers:', ort.get_available_providers()); print('OpenCV:', cv2.__version__, 'NumPy:', np.__version__)"

# --- Your app ---
WORKDIR /app
COPY app/main.py /app/main.py

# Entry (expects VIDEO_URL and IMAGE_URL env vars)
ENTRYPOINT ["python3.11", "/app/main.py"]