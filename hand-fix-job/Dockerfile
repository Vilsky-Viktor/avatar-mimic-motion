# Use a modern CUDA 12.1 base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    wget \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir wheel==0.43.0

# Install PyTorch (CUDA 12.1) - Initial Installation
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- Install MeshGraphormer Code & MOST Dependencies ---

# 3. Clone MeshGraphormer repo
RUN git clone https://github.com/microsoft/MeshGraphormer.git $APP_HOME/MeshGraphormer

# Clone its vendored dependencies locally
WORKDIR $APP_HOME/MeshGraphormer
RUN git clone https://github.com/huggingface/transformers.git \
    && cd transformers \
    && git checkout 067923d
RUN git clone https://github.com/hassony2/manopth.git \
    && cd manopth \
    && git checkout 4f1dcad
WORKDIR $APP_HOME # Return to app root

# 4. Install MeshGraphormer-specific requirements
RUN pip install --no-cache-dir ninja==1.11.1.1
RUN pip install --no-cache-dir --no-build-isolation \
    mmpose==1.3.1 \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html \
    onnxruntime-gpu==1.18.0
RUN pip install --no-cache-dir \
    yacs==0.1.8 \
    termcolor==2.4.0 \
    paramiko==3.4.0 \
    matplotlib==3.8.4 \
    easydict==1.13 \
    timm==0.9.16 \
    einops==0.8.0 \
    h5py==3.11.0 \
    boto3 \
    nltk \
    cython

# 5. Install manotorch (separate from MeshGraphormer's vendored copy)
RUN pip install --no-cache-dir git+https://github.com/lixiny/manotorch

# --- END Partial MeshGraphormer ---

# Install core SDXL, ControlNet, etc.
RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.40.1 \
    accelerate==0.30.1 \
    "opencv-python-headless==4.9.0.80"

# Install SAM2 (Segment Anything 2)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

RUN python -c "import sam2; print('SAM2 imported successfully during build!')"

# --- Check/Reinstall PyTorch stack AFTER SAM2 ---
RUN python -c "import torch; exit(0) if torch.__version__.startswith('2.1.2') else exit(1)" || \
    (echo 'PyTorch version changed, force-reinstalling 2.1.2...' && \
     pip install --no-cache-dir --force-reinstall \
     torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
     --index-url https://download.pytorch.org/whl/cu121)

# --- Install PyTorch3D LAST ---
RUN pip install --no-cache-dir fvcore==0.1.5.post20221221 iopath==0.1.10
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

# --- Install other core dependencies ---
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    ultralytics \
    scipy \
    google-cloud-storage==2.17.0 \
    google-cloud-aiplatform==1.56.0 \
    gcsfs==2024.6.0 \
    tqdm==4.66.4 \
    PyYAML==6.0.1 \
    imageio==2.34.1 \
    imageio-ffmpeg==0.5.1 \
    Pillow==10.3.0 \
    huggingface_hub==0.25.0

# --- Setup PYTHONPATH ---
ENV PYTHONPATH="/app/MeshGraphormer/manopth:/app/MeshGraphormer:/app:${PYTHONPATH:-}"

# ---
# --- THIS IS THE FIX ---
# ---
# Create the *correct* local directories that main.py will download models into
RUN mkdir -p $APP_HOME/MeshGraphormer/models/graphormer_release
RUN mkdir -p $APP_HOME/src/modeling/data
RUN mkdir -p $APP_HOME/models/hrnet
RUN mkdir -p $APP_HOME/checkpoints
RUN mkdir -p $APP_HOME/configs/sam2.1
RUN mkdir -p /app/MeshGraphormer/src/modeling/bert/bert-base-uncased
# --- END FIX ---

# Copy the job source code into the container
COPY hand_fix_job ./hand_fix_job

# WORKDIR is still /app, which is correct
# Set the entrypoint for Vertex AI
ENTRYPOINT ["python", "-m", "hand_fix_job.main"]