FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl ca-certificates \
    python3.11 python3.11-dev python3.11-distutils \
    build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Poetry
RUN pip install poetry

# Copy project
COPY pyproject.toml /app/
COPY avatar_training_job /app/avatar_training_job

# Install deps
RUN poetry config virtualenvs.create false && poetry install --no-root

# Defaults (override in Vertex job)
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV PHOTOS_PATH=""
ENV POSES_PATH=""
ENV AI_AVATAR_ID=""

ENTRYPOINT ["python3.11", "-m", "avatar_training_job.main"]