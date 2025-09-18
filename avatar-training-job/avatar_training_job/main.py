import os
import logging
from pathlib import Path
from google.cloud import storage
from avatar_training_job.trainer import train_lora

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("avatar-training")

# --- ENV VARS ---
JOBS_BUCKET = os.environ.get("JOBS_BUCKET", "billion-ai-girls-jobs")
PHOTOS_PATH = os.environ["PHOTOS_PATH"]  # e.g. ai_avatars/{ai_avatar_id}/photos/
POSES_PATH = os.environ["POSES_PATH"]    # e.g. ai_avatars/{ai_avatar_id}/dwpose/
AI_AVATAR_ID = os.environ["AI_AVATAR_ID"]
LOCAL_ROOT = Path("/app/data")
OUTPUT_DIR = LOCAL_ROOT / "output"
MODELS_DIR = LOCAL_ROOT / "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1/unet"

# --- GCS Helpers ---
def download_from_gcs(bucket_name, prefix, local_dir: Path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    local_dir.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        dest = local_dir / Path(blob.name).name
        log.info(f"Downloading {blob.name} -> {dest}")
        blob.download_to_filename(dest)

def upload_to_gcs(bucket_name, src_dir: Path, dst_prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for file in src_dir.glob("*"):
        blob = bucket.blob(f"{dst_prefix}/{file.name}")
        blob.upload_from_filename(file)
        log.info(f"Uploaded {file} -> gs://{bucket_name}/{dst_prefix}/{file.name}")

def main():
    photos_dir = LOCAL_ROOT / "photos"
    poses_dir = LOCAL_ROOT / "poses"

    log.info("Downloading photos...")
    download_from_gcs(JOBS_BUCKET, PHOTOS_PATH, photos_dir)

    log.info("Downloading DWPose keypoints...")
    download_from_gcs(JOBS_BUCKET, POSES_PATH, poses_dir)

    log.info("Downloading base UNet...")
    download_from_gcs(JOBS_BUCKET, "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1/unet", MODELS_DIR)

    log.info("Starting LoRA training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lora_path = OUTPUT_DIR / "lora-mimicmotion-unet"

    train_lora(unet_dir=MODELS_DIR, photos_dir=photos_dir, poses_dir=poses_dir, out_dir=lora_path)

    dst = f"ai_avatars/{AI_AVATAR_ID}/lora"
    upload_to_gcs(JOBS_BUCKET, lora_path, dst)

if __name__ == "__main__":
    main()