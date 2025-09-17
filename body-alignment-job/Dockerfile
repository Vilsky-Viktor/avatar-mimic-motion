FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.10 python3.10-distutils python3.10-dev \
    build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# pip + poetry
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN curl -sSL https://install.python-poetry.org | python3.10 -
ENV PATH="/root/.local/bin:${PATH}"

# --- clone official repo ---
RUN git clone https://github.com/SMPLCap/SMPLest-X.git /app/SMPLest-X

# --- install repo deps (kept in its requirements.txt / scripts) ---
# We let the repo control its own pins; modern Torch wheels for CUDA 12.1 are fine.
RUN bash -lc "cd /app/SMPLest-X && \
    if [ -f scripts/install.sh ]; then bash scripts/install.sh; fi; \
    if [ -f requirements.txt ]; then python3.10 -m pip install -r requirements.txt; fi"

# --- our job deps via Poetry (only small extras) ---
COPY pyproject.toml /app/
RUN poetry config virtualenvs.create false && poetry install --no-root

# --- job code ---
COPY body_alignment_job /app/body_alignment_job

# envs (override at runtime)
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV SMPlEST_MODEL_DIR=smplest_x_h
ENV FPS=4

ENTRYPOINT ["python3.10", "-m", "body_alignment_job.main"]