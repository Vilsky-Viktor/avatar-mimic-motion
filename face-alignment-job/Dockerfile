FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip build-essential \
    libgl1 libglib2.0-0 ffmpeg pkg-config \
    libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
    libswscale-dev libavutil-dev \
    python3.9 python3.9-venv python3.9-distutils python3.9-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Pip for Python 3.9 ---
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# --- Poetry (with Python 3.9) ---
RUN python3.9 -m pip install --no-cache-dir --upgrade pip && \
    python3.9 -m pip install --no-cache-dir "poetry==1.8.3"

# --- Verify ---
RUN python3.9 --version && python3.9 -m pip --version && poetry --version

# --- Clone and install Spectre ---
RUN git clone https://github.com/filby89/spectre /app/spectre && \
    sed -i 's/^av==9\.2\.0/av==10.0.0/' /app/spectre/requirements.txt && \
    python3.9 -m pip install --no-cache-dir -r /app/spectre/requirements.txt

COPY pyproject.toml poetry.lock* /app/
COPY face_alignment_job /app/face_alignment_job

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN python3.9 -m pip uninstall -y certifi || true && \
    echo "import os\n\
from pathlib import Path\n\
def where():\n\
    return os.environ.get('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt')\n" \
    > /usr/local/lib/python3.9/dist-packages/certifi.py

RUN poetry config virtualenvs.create false && \
    poetry export -f requirements.txt --without-hashes | python3.9 -m pip install --no-cache-dir -r /dev/stdin

RUN python3.9 -m pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3.9 -m pip install --no-cache-dir \
    pytorch3d==0.7.2 \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/spectre:${PYTHONPATH}"
ENV SPECTRE_MODELS_DIR=/models/spectre

ENTRYPOINT ["python3.9", "-m", "face_alignment_job.main"]