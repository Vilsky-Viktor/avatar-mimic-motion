import os
import logging
import subprocess
from pathlib import Path

from mimic_motion.gcs_utils import download_folder, download_blob, upload_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("wan-2-2")

# --- Environment Variables ---
JOBS_BUCKET = os.environ["JOBS_BUCKET"]
EXECUTION_ID = os.environ["EXECUTION_ID"]
REF_IMAGE_PATH = os.environ["REF_IMAGE_PATH"]
REF_VIDEO_PATH = os.environ["REF_VIDEO_PATH"]
MODE = os.environ.get("MODE", "retarget").lower()
OUTPUT_FOLDER_NAME = os.environ.get("OUTPUT_FOLDER_NAME", "output")

# --- Quality Tuning Parameters for the Animate Flow ---
SAMPLING_STEPS = os.environ.get("SAMPLING_STEPS", "40")
GUIDE_SCALE = os.environ.get("GUIDE_SCALE", "1.0")
SAMPLE_SHIFT = os.environ.get("SAMPLE_SHIFT", "5.0")
SAMPLE_SOLVER = os.environ.get("SAMPLE_SOLVER", "unipc")
REFERT_NUM = os.environ.get("REFERT_NUM", "1")
WIDTH = os.environ.get("WIDTH", "720")
HEIGHT = os.environ.get("HEIGHT", "1280")


# --- Path Definitions ---
REPO = Path("/app/Wan2.2")
LOCAL_ROOT = Path("/app/local")
INPUT_DIR = LOCAL_ROOT / "input"
OUTPUT_DIR = LOCAL_ROOT / OUTPUT_FOLDER_NAME

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

    log.info("Downloading wan2.2 models...")
    download_folder(JOBS_BUCKET, str(CHECKPOINT_BUCKET_PATH), str(CHECKPOINT_LOCAL_PATH))

    # --- Animate Preprocessing for Motion Mimicry ---
    preprocess_cmd = [
        "python", "./wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", str(PROCESS_CHECKPOINT_LOCAL_PATH),
        "--video_path", str(ref_vid),
        "--refer_path", str(ref_img),
        "--save_path", str(OUTPUT_DIR),
        "--resolution_area", WIDTH, HEIGHT,
    ]
    if MODE == "retarget":
        preprocess_cmd += ["--retarget_flag", "--use_flux"]
    else: # replace
        preprocess_cmd += ["--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1", "--replace_flag"]

    log.info("Running animate preprocessing…")
    run(preprocess_cmd, REPO)

    # --- Generate with Animate Flow ---
    generate_cmd = [
        "python", "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", str(CHECKPOINT_LOCAL_PATH),
        "--src_root_path", str(OUTPUT_DIR),
        "--refert_num", REFERT_NUM,
        "--save_file", str(RESULT_MP4_FILE_PATH),
        "--sample_solver", SAMPLE_SOLVER,
        "--sample_steps", SAMPLING_STEPS,
        "--sample_guide_scale", GUIDE_SCALE,
        "--sample_shift", SAMPLE_SHIFT,
    ]

    if MODE == "retarget":
        generate_cmd += ["--use_relighting_lora"]
    if MODE == "replace":
        generate_cmd += ["--replace_flag", "--use_relighting_lora"]

    log.info("Running WAN 2.2 animate inference…")
    run(generate_cmd, REPO)

    # Upload the whole OUTPUT_DIR
    dst_prefix = f"jobs/{EXECUTION_ID}/output"
    uploaded = upload_folder(JOBS_BUCKET, OUTPUT_DIR, dst_prefix)
    log.info("Uploaded %d files to gs://%s/%s/", uploaded, JOBS_BUCKET, dst_prefix.rstrip("/"))

if __name__ == "__main__":
    main()

