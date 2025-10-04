import os
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml
from mimic_motion_job.gcs_utils import download_folder, download_blob, upload_blob

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mimic-motion-job")

JOBS_BUCKET = os.environ["JOBS_BUCKET"]
EXECUTION_ID = os.environ["EXECUTION_ID"]
REF_IMAGE_PATH = os.environ["REF_IMAGE_PATH"]
REF_VIDEO_PATH = os.environ["REF_VIDEO_PATH"]
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
LORA_SCALE = float(os.getenv("LORA_SCALE", "1.8"))

NUM_FRAMES = int(os.getenv("NUM_FRAMES", "72"))
RESOLUTION = int(os.getenv("RESOLUTION", "576"))
FRAMES_OVERLAP = int(os.getenv("FRAMES_OVERLAP", "6"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "25"))
NOISE_AUG_STRENGTH = int(os.getenv("NOISE_AUG_STRENGTH", "0"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "2.0"))
SAMPLE_STRIDE = int(os.getenv("SAMPLE_STRIDE", "2"))
FPS = int(os.getenv("FPS", "15"))
SEED = int(os.getenv("SEED", "42"))

REPO = Path("/app/MimicMotion")
BASE_CONFIG = REPO / "configs" / "test.yaml"
MODELS_PATH = REPO / "models"

LOCAL_ROOT = Path("/app/local")
INPUT_DIR = LOCAL_ROOT / "input"
OUTPUT_DIR = LOCAL_ROOT / "output"
CONFIGS_DIR = LOCAL_ROOT / "configs"

def write_yaml(cfg: dict[str, Any], dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        # IMPORTANT: keep original order & content; do not sort keys
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    ref_img = INPUT_DIR / "reference.jpg"
    ref_vid = INPUT_DIR / "driving.mp4"
    log.info("Downloading inputs from GCS…")
    download_blob(JOBS_BUCKET, REF_IMAGE_PATH, ref_img)
    download_blob(JOBS_BUCKET, REF_VIDEO_PATH, ref_vid)

    config = {
        "base_model_path": "/app/models/stable-video-diffusion-img2vid-xt-1-1",
        "lora_path": "/app/MimicMotion/models/lora.safetensors",
        "lora_alpha": LORA_ALPHA,
        "lora_scale": LORA_SCALE,
        "ckpt_path": "models/MimicMotion_1-1.pth",
        "test_case": [
            {
                "ref_video_path": str(ref_vid),
                "ref_image_path": str(ref_img),
                "num_frames": NUM_FRAMES,
                "resolution": RESOLUTION,
                "frames_overlap": FRAMES_OVERLAP,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "noise_aug_strength": NOISE_AUG_STRENGTH,
                "guidance_scale": GUIDANCE_SCALE,
                "sample_stride": SAMPLE_STRIDE,
                "fps": FPS,
                "seed": SEED
            }
        ]
    }

    log.info(config)

    cfg_path = CONFIGS_DIR / "job_test.yaml"
    write_yaml(config, cfg_path)
    log.info(f"Wrote config (based on repo test.yaml) to: {cfg_path}")

    log.info("downloading models ...")
    download_blob(JOBS_BUCKET, "models/mimic_motion/MimicMotion_1-1.pth", MODELS_PATH / "MimicMotion_1-1.pth")
    download_blob(JOBS_BUCKET, "models/mimic_motion/dw-ll_ucoco_384.onnx", MODELS_PATH / "DWPose" / "dw-ll_ucoco_384.onnx")
    download_blob(JOBS_BUCKET, "models/mimic_motion/yolox_l.onnx", MODELS_PATH / "DWPose" / "yolox_l.onnx")
    download_folder(JOBS_BUCKET, "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1/", "/app/models/stable-video-diffusion-img2vid-xt-1-1")

    download_blob(JOBS_BUCKET, "ai_avatars/ver1/models/lora_mimicmotion_unet/final.safetensors", MODELS_PATH / "lora.safetensors")

    # Run inference exactly like README:
    #   python inference.py --inference_config configs/test.yaml
    # Here we pass our generated file path.
    cmd = [
        "python3.11",
        "inference.py",
        "--inference_config",
        str(cfg_path),
        "--output_dir",
        str(OUTPUT_DIR)
    ]
    log.info("Running MimicMotion inference…")
    subprocess.run(cmd, cwd=str(REPO), check=True)

    # Find the output video (repo writes under output_dir)
    mp4 = None
    for p in OUTPUT_DIR.rglob("*.mp4"):
        mp4 = p
        break
    if not mp4:
        raise FileNotFoundError(f"No .mp4 produced under {OUTPUT_DIR}")

    dst = f"jobs/{EXECUTION_ID}/generated.mp4"
    upload_blob(JOBS_BUCKET, mp4, dst, content_type="video/mp4")
    log.info(f"Uploaded: gs://{JOBS_BUCKET}/{dst}")

if __name__ == "__main__":
    main()