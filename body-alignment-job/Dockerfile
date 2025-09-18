FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.9 python3.9-distutils python3.9-dev \
    build-essential libgl1 libglib2.0-0 \
    libegl1 libgles2 \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dri mesa-common-dev \
    && rm -rf /var/lib/apt/lists/*

# ensure python points to python3.9
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# pip + poetry (Python 3.9 uses the normal bootstrap)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN curl -sSL https://install.python-poetry.org | python3.9 -
ENV PATH="/root/.local/bin:${PATH}"

# --- clone official repo ---
RUN git clone https://github.com/SMPLCap/SMPLest-X.git /app/SMPLest-X

# --- install PyTorch (CUDA 11.3 build for Python 3.9) ---
RUN python3.9 -m pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# --- install repo deps ---
RUN python3.9 -m pip install -r /app/SMPLest-X/requirements.txt

# --- our job deps via Poetry ---
COPY pyproject.toml poetry.lock* /app/

RUN poetry export -f requirements.txt --without-hashes \
    | python3.9 -m pip install --no-cache-dir -r /dev/stdin && \
    python3.9 -m pip install google-cloud-storage==2.18.2

# --- OpenGL Python bindings ---
RUN python3.9 -m pip install pyopengl==3.1.4

# --- job code ---
COPY body_alignment_job /app/body_alignment_job

# envs (override at runtime)
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV SMPlEST_MODEL_DIR=smplest_x_h
ENV FPS=4
ENV PYOPENGL_PLATFORM=egl
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ENTRYPOINT ["python3.9", "-m", "body_alignment_job.main"]