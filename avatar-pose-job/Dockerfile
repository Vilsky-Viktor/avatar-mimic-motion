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

# Clone DWPose
RUN git clone https://github.com/IDEA-Research/DWPose.git /app/DWPose

# Copy job
COPY pyproject.toml /app/
COPY avatar_pose_job /app/avatar_pose_job

# Install deps
RUN poetry config virtualenvs.create false && poetry install --no-root

# Download pretrained checkpoint (large model as example)
RUN curl -L -o /app/DWPose/dwpose-l.pth https://huggingface.co/IDEA-Research/DWPose/resolve/main/dwpose-l.pth

# Defaults
ENV JOBS_BUCKET=billion-ai-girls-jobs
ENV AI_AVATAR_ID=""

ENTRYPOINT ["python3.11", "-m", "avatar_pose_job.main"]