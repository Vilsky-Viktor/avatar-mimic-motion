import os
import logging
import subprocess
from pathlib import Path

from mimic_motion.gcs_utils import download_folder, download_blob, upload_blob, upload_folder  # UPDATED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("wan-2-2")

JOBS_BUCKET = os.environ["JOBS_BUCKET"]
EXECUTION_ID = os.environ["EXECUTION_ID"]
REF_IMAGE_PATH = os.environ["REF_IMAGE_PATH"]
REF_VIDEO_PATH = os.environ["REF_VIDEO_PATH"]
MODE = os.environ["MODE"].lower()

# Validate & coerce
try:
    WIDTH = int(os.environ["WIDTH"])
    HEIGHT = int(os.environ["HEIGHT"])
except (KeyError, ValueError) as e:
    raise ValueError("WIDTH and HEIGHT environment variables must be set to integers") from e

REPO = Path("/app/Wan2.2")
LOCAL_ROOT = Path("/app/local")
INPUT_DIR = LOCAL_ROOT / "input"
OUTPUT_DIR = LOCAL_ROOT / "output"

CHECKPOINT_BUCKET_PATH = "models/wan-2-2/animate-14b/"
CHECKPOINT_LOCAL_PATH = Path("/app/models/animate-14b")
PROCESS_CHECKPOINT_LOCAL_PATH = CHECKPOINT_LOCAL_PATH / "process_checkpoint"
RESULT_MP4_FILE_PATH = OUTPUT_DIR / f"generated.mp4"

def run(cmd: list[str], cwd: Path) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)

    ref_img = INPUT_DIR / "reference.png"
    ref_vid = INPUT_DIR / "driving.mp4"

    log.info("Downloading inputs from GCS…")
    download_blob(JOBS_BUCKET, REF_IMAGE_PATH, ref_img)
    download_blob(JOBS_BUCKET, REF_VIDEO_PATH, ref_vid)

    log.info("Downloading models…")
    download_folder(JOBS_BUCKET, str(CHECKPOINT_BUCKET_PATH), str(CHECKPOINT_LOCAL_PATH))

    if MODE not in {"retarget", "replace"}:
        raise ValueError("MODE must be either 'retarget' or 'replace'")

    # --- Preprocess ---
    preprocess_cmd = [
        "python", "./wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", str(PROCESS_CHECKPOINT_LOCAL_PATH),
        "--video_path", str(ref_vid),
        "--refer_path", str(ref_img),
        "--save_path", str(OUTPUT_DIR),
        "--resolution_area", str(WIDTH), str(HEIGHT),
    ]
    if MODE == "retarget":
        preprocess_cmd += ["--retarget_flag", "--use_flux"]
    else:  # replace
        preprocess_cmd += ["--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1", "--replace_flag"]

    log.info("Running preprocessing…")
    run(preprocess_cmd, REPO)

    # --- Generate ---
    generate_cmd = [
        "python", "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", str(CHECKPOINT_LOCAL_PATH),
        "--src_root_path", str(OUTPUT_DIR),
        "--refert_num", "1",
        "--size", f"{str(WIDTH)}*{str(HEIGHT)}",
        "--save_file", str(RESULT_MP4_FILE_PATH)
    ]
    if MODE == "replace":
        generate_cmd += ["--replace_flag", "--use_relighting_lora"]

    log.info("Running WAN 2.2 inference…")
    run(generate_cmd, REPO)

    # Upload the whole OUTPUT_DIR
    dst_prefix = f"jobs/{EXECUTION_ID}/output"
    uploaded = upload_folder(JOBS_BUCKET, OUTPUT_DIR, dst_prefix)
    log.info("Uploaded %d files to gs://%s/%s/", uploaded, JOBS_BUCKET, dst_prefix.rstrip("/"))

if __name__ == "__main__":
    main()