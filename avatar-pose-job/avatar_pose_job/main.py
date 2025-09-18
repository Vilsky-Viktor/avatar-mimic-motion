import os
import logging
from pathlib import Path
from google.cloud import storage
from avatar_pose_job.dwpose_infer import run_dwpose_on_dir

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("avatar-dwpose")

# --- ENV VARS ---
JOBS_BUCKET = os.environ.get("JOBS_BUCKET", "billion-ai-girls-jobs")
AI_AVATAR_ID = os.environ["AI_AVATAR_ID"]
LOCAL_ROOT = Path("/app/data")

INPUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/identity_images"
OUTPUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/dwpose_keypoints"

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

def upload_to_gcs(bucket_name, local_dir: Path, prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for file in local_dir.glob("*.json"):
        dst = f"{prefix}/{file.name}"
        blob = bucket.blob(dst)
        blob.upload_from_filename(file)
        log.info(f"Uploaded {file} -> gs://{bucket_name}/{dst}")

def main():
    photos_dir = LOCAL_ROOT / "identity_images"
    out_dir = LOCAL_ROOT / "dwpose_keypoints"

    log.info("Downloading photos...")
    download_from_gcs(JOBS_BUCKET, INPUT_PREFIX, photos_dir)

    log.info("Running DWPose inference...")
    run_dwpose_on_dir(photos_dir, out_dir)

    log.info("Uploading keypoints...")
    upload_to_gcs(JOBS_BUCKET, out_dir, OUTPUT_PREFIX)

if __name__ == "__main__":
    main()