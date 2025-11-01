# Use the specific NVIDIA CUDA base image recommended by the RIFE repository
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies. This uses the standard, reliable ffmpeg from the Ubuntu repos.
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set python3.8 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Set the working directory
WORKDIR /app

# Clone the newer ECCV2022 RIFE repository
RUN git clone https://github.com/hzwer/ECCV2022-RIFE.git

# Set the working directory to the cloned repo
WORKDIR /app/ECCV2022-RIFE

# Use `sed` to modify requirements.txt in-place to prevent dependency conflicts.
RUN sed -i 's/moviepy>=1.0.3/moviepy==1.0.3/' requirements.txt

# Install all Python dependencies in a single, clean layer.
RUN pip install -r requirements.txt && \
    pip install gcsfs google-cloud-storage cupy-cuda11x

# Copy your job script into the container
WORKDIR /app
COPY interpolation_job /app/interpolation_job

# Set the command to run when the container starts
CMD ["python3", "-m", "interpolation_job.main"]

