FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip build-essential gnupg2 ca-certificates \
    libgl1 libglib2.0-0 ffmpeg python3 python3-pip python3-venv && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install poetry==$POETRY_VERSION

COPY pyproject.toml poetry.lock* ./

COPY preprocess_image_job /app/preprocess_image_job

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

RUN pip3 install git+https://github.com/facebookresearch/segment-anything-2.git

RUN pip3 uninstall -y opencv-python opencv-contrib-python || true && \
    pip3 install --no-cache-dir opencv-python-headless==4.10.0.84 && \
    pip3 install tf-keras

RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

ENV PYTHONUNBUFFERED=1
ENV SAM2_DIR=/models/sam2

ENTRYPOINT ["python3", "-m", "preprocess_image_job.main"]