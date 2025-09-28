import os
import logging
from pathlib import Path
from google.cloud import storage
from avatar_pose_job.infer import run_dwpose_on_dir

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("avatar-dwpose")

# --- ENV VARS ---
JOBS_BUCKET = os.environ.get("JOBS_BUCKET", "billion-ai-girls-jobs")
AI_AVATAR_ID = os.environ["AI_AVATAR_ID"]
DWPOSE_PATH = Path("/app/DWPose")
LOCAL_ROOT = Path("/app/data")

INPUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/body_crops"
OUTPUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/pose_keypoints"

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

def download_blob(bucket: str, src: str, dst: Path):
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(src)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{src} not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))

def upload_to_gcs(bucket_name: str, local_dir: Path, prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = list(local_dir.glob("*.json")) + list(local_dir.glob("*.png"))

    if not files:
        log.warning(f"No files found in {local_dir}")
        return

    for file in files:
        dst = f"{prefix}/{file.name}"
        blob = bucket.blob(dst)
        blob.upload_from_filename(file)
        log.info(f"Uploaded {file} -> gs://{bucket_name}/{dst}")

def main():
    photos_dir = LOCAL_ROOT / "identity_images"
    out_dir = LOCAL_ROOT / "dwpose_keypoints"

    log.info("Downloading photos...")
    download_from_gcs(JOBS_BUCKET, INPUT_PREFIX, photos_dir)

    log.info("Downloading model...")
    download_blob(JOBS_BUCKET, "models/dwpose/dw-ll_ucoco_384.onnx", DWPOSE_PATH / "models" / "dw-ll_ucoco_384.onnx")

    log.info("Running DWPose inference...")
    run_dwpose_on_dir(photos_dir, out_dir)

    log.info("Uploading keypoints...")
    upload_to_gcs(JOBS_BUCKET, out_dir, OUTPUT_PREFIX)

if __name__ == "__main__":
    main()