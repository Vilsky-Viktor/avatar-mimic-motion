import os
import logging
import subprocess
from pathlib import Path
from google.cloud import storage
from typing import Tuple
import cv2

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rife-interpolation-job")

# --- Path Definitions ---
RIFE_REPO_PATH = Path("/app/ECCV2022-RIFE")
LOCAL_INPUT_DIR = Path("/app/input")
LOCAL_OUTPUT_DIR = Path("/app/output")
MODEL_DIR = RIFE_REPO_PATH / "train_log"
MODEL_FILE = MODEL_DIR / "flownet.pkl"

def parse_gcs_url(gcs_url: str) -> Tuple[str, str]:
    """Parses a GCS URL into bucket name and blob name."""
    if not gcs_url.startswith("gs://"):
        raise ValueError("Invalid GCS URL. Must start with 'gs://'")
    
    parts = gcs_url[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError("Invalid GCS URL format. Expected 'gs://<bucket>/<blob_path>'")
        
    bucket_name = parts[0]
    blob_name = parts[1]
    return bucket_name, blob_name

def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: Path):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    log.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_name}...")
    destination_file_name.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination_file_name))
    log.info("Download complete.")

def upload_blob(bucket_name: str, source_file_name: Path, destination_blob_name: str):
    """Uploads a single file to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    log.info(f"Uploading {source_file_name} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(str(source_file_name))
    log.info("Upload complete.")

def run_subprocess(command: list, error_message: str):
    """Runs a subprocess command, checks for errors, and logs output."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=RIFE_REPO_PATH
        )
        log.info(f"Successfully ran: {' '.join(command)}")
        if result.stdout: log.info(f"STDOUT: {result.stdout}")
        if result.stderr: log.warning(f"STDERR: {result.stderr}")
    except subprocess.CalledProcessError as e:
        log.error(f"FATAL: {error_message}")
        log.error(f"Return Code: {e.returncode}")
        log.error(f"STDOUT: {e.stdout}")
        log.error(f"STDERR: {e.stderr}")
        raise

def get_video_fps(video_path: Path) -> float:
    """Gets the FPS of a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Could not open video file {video_path} to determine FPS. Returning 30.0 as default.")
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def main():
    """
    Main function for the Vertex AI job.
    - Downloads a source video from GCS.
    - Runs RIFE video interpolation.
    - Uploads the output video back to the same GCS folder.
    """
    # --- Get Environment Variables ---
    video_url = os.environ.get("VIDEO_URL")
    if not video_url:
        raise ValueError("VIDEO_URL must be provided.")

    exp_factor = int(os.environ.get("EXP", "1"))
    scale = os.environ.get("SCALE", "1.0")

    log.info(f"Received video URL: {video_url}")
    log.info(f"Interpolation exponent (exp): {exp_factor}")
    log.info(f"Quality/speed scale: {scale}")

    try:
        bucket_name, source_blob_name = parse_gcs_url(video_url)
    except ValueError as e:
        raise ValueError(f"FATAL: {e}")

    rife_model_gcs_path = "models/rife/flownet.pkl"

    # --- Prepare Local Environment ---
    LOCAL_INPUT_DIR.mkdir(exist_ok=True)
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    source_video_path = Path(source_blob_name)
    local_input_path = LOCAL_INPUT_DIR / source_video_path.name

    # --- Download Model and Source Video ---
    try:
        download_blob(bucket_name, rife_model_gcs_path, MODEL_FILE)
        download_blob(bucket_name, source_blob_name, local_input_path)
    except Exception as e:
        log.error(f"FATAL: Failed during download. Error: {e}")
        raise

    # --- RIFE Interpolation ---
    log.info("Starting RIFE video interpolation...")
    original_fps = get_video_fps(local_input_path)
    interpolation_multiplier = 2 ** exp_factor
    target_fps = original_fps * interpolation_multiplier
    log.info(f"Original FPS: {original_fps:.2f}, Target FPS: {target_fps:.2f}")

    output_filename = f"{source_video_path.stem}_interpolated.mp4"
    local_output_path = LOCAL_OUTPUT_DIR / output_filename
    
    rife_command = [
        "python3", "inference_video.py",
        "--exp", str(exp_factor), "--scale", scale,
        "--fps", str(int(round(target_fps))),
        "--video", str(local_input_path),
        "--output", str(local_output_path),
        "--model", str(MODEL_DIR)
    ]
    run_subprocess(rife_command, "RIFE interpolation failed.")

    # --- Upload Result File ---
    if not local_output_path.exists():
        raise FileNotFoundError(f"RIFE did not produce the output file: {local_output_path}")
    
    destination_blob_name = str(source_video_path.parent / output_filename)
    
    log.info(f"Uploading result to gs://{bucket_name}/{destination_blob_name}...")
    upload_blob(bucket_name, local_output_path, destination_blob_name)
    log.info("Job finished successfully.")

if __name__ == "__main__":
    main()

