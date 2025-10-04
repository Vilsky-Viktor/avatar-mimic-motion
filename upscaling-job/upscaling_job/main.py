import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
from tqdm import tqdm
import torch
from google.cloud import storage

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# -----------------------------
# Env / IO
# -----------------------------
BUCKET = os.environ.get("JOBS_BUCKET", "")
JOB_FOLDER = os.environ.get("JOB_FOLDER", "").strip().strip("/")
INPUT_URI = os.environ.get("INPUT_URI", "").strip()  # optional gs://
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_realesrgan_x2")
ESRGAN_WEIGHTS_URI = os.environ.get("ESRGAN_WEIGHTS_URI", "").strip()  # gs://.../RealESRGAN_x2plus.pth

SCALE = int(os.environ.get("SCALE", "2"))
TILE = int(os.environ.get("TILE", "256"))
FP16 = os.environ.get("FP16", "1") == "1"
CRF = os.environ.get("CRF", "18")
PRESET = os.environ.get("PRESET", "slow")

LOCAL_DIR = Path("/app/local")
FRAMES_DIR = LOCAL_DIR / "frames"
UPSCALED_DIR = LOCAL_DIR / "frames_up"
INPUT_PATH = LOCAL_DIR / "input.mp4"
OUTPUT_PATH = LOCAL_DIR / "output_up.mp4"
WEIGHTS_DIR = Path("/app/models")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


def gcs_client():
    return storage.Client()


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def pick_latest_mp4(bucket: storage.Bucket, prefix: str) -> Optional[storage.Blob]:
    blobs = list(bucket.list_blobs(prefix=prefix))
    mp4s = [b for b in blobs if b.name.lower().endswith(".mp4")]
    mp4s = [b for b in mp4s if OUTPUT_SUFFIX not in Path(b.name).stem]  # skip previous outputs
    if not mp4s:
        return None
    mp4s.sort(key=lambda b: b.updated, reverse=True)
    return mp4s[0]


def download_blob(bkt: storage.Bucket, blob_name: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading gs://{bkt.name}/{blob_name} -> {dest}")
    bkt.blob(blob_name).download_to_filename(str(dest))


def upload_blob(bkt: storage.Bucket, src: Path, dest_blob: str, content_type="video/mp4"):
    log(f"Uploading {src} -> gs://{bkt.name}/{dest_blob}")
    b = bkt.blob(dest_blob)
    b.content_type = content_type
    b.upload_from_filename(str(src))


def extract_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 25.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 1e-3:
        fps = 25.0
    return float(fps)


def extract_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total, desc="Decode frames")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame)
        pbar.update(1)
    cap.release()
    pbar.close()
    return idx


def build_upsampler(device: str, half: bool, scale: int, weights_path: Path) -> RealESRGANer:
    log(f"Creating Real-ESRGAN (x{scale}) half={half} weights={weights_path}")
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights_path),
        model=model,
        tile=TILE,
        tile_pad=10,
        pre_pad=0,
        half=half and (device == "cuda"),
        device=torch.device(device)
    )
    return upsampler


def ensure_weights() -> Path:
    local = WEIGHTS_DIR / "RealESRGAN_x2plus.pth"
    if local.exists():
        return local
    if ESRGAN_WEIGHTS_URI:
        wbkt, wkey = parse_gs_uri(ESRGAN_WEIGHTS_URI)
        log(f"Downloading Real-ESRGAN weights from {ESRGAN_WEIGHTS_URI}")
        storage.Client().bucket(wbkt).blob(wkey).download_to_filename(str(local))
        return local
    # Fallback: download officially (requires egress)
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth"
    log(f"Downloading Real-ESRGAN weights from {url}")
    import urllib.request, shutil as _shutil
    with urllib.request.urlopen(url) as r, open(local, "wb") as f:
        _shutil.copyfileobj(r, f)
    return local


def upscale_frames(upsampler: RealESRGANer, frames_dir: Path, out_dir: Path, scale: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.png"))
    pbar = tqdm(frames, desc=f"Upscale x{scale}")
    for fn in pbar:
        img = cv2.imread(str(fn), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read frame: {fn}")
        output, _ = upsampler.enhance(img, outscale=scale)
        cv2.imwrite(str(out_dir / fn.name), output)
    pbar.close()


def encode_video(up_dir: Path, fps: float, out_path: Path):
    pattern = str(up_dir / "frame_%06d.png")
    # Produce a **silent** MP4 explicitly (-an)
    cmd = (
        f'ffmpeg -y -r {fps:.6f} -i "{pattern}" '
        f'-c:v libx264 -pix_fmt yuv420p -crf {CRF} -preset {PRESET} -an -movflags +faststart "{out_path}"'
    )
    rc = os.system(cmd)
    if rc != 0 or not out_path.exists():
        raise RuntimeError("ffmpeg encode failed")


def main():
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device={device}  FP16={FP16}  SCALE={SCALE}  TILE={TILE}")

    assert BUCKET, "Set JOBS_BUCKET"
    client = gcs_client()
    bucket = client.bucket(BUCKET)

    # Pick input
    if INPUT_URI:
        in_bkt, in_key = parse_gs_uri(INPUT_URI)
        assert in_bkt == BUCKET, "Set JOBS_BUCKET to match INPUT_URI bucket"
        input_blob = bucket.blob(in_key)
        if not input_blob.exists():
            raise FileNotFoundError(f"{INPUT_URI} not found")
    else:
        assert JOB_FOLDER, "Set JOB_FOLDER (prefix within JOBS_BUCKET) or INPUT_URI"
        input_blob = pick_latest_mp4(bucket, JOB_FOLDER + "/")
        if input_blob is None:
            raise FileNotFoundError(f"No MP4 found under gs://{BUCKET}/{JOB_FOLDER}")

    input_name = Path(input_blob.name).name
    input_stem = Path(input_name).stem
    output_blob_name = f"{Path(input_blob.name).parent}/{input_stem}{OUTPUT_SUFFIX}.mp4"

    # Clean local
    if LOCAL_DIR.exists():
        shutil.rmtree(LOCAL_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    UPSCALED_DIR.mkdir(parents=True, exist_ok=True)

    # Download input
    download_blob(bucket, input_blob.name, INPUT_PATH)

    # FPS -> frames -> upsample -> encode (silent)
    fps = extract_fps(INPUT_PATH)
    log(f"FPS={fps:.4f}")
    n = extract_frames(INPUT_PATH, FRAMES_DIR)
    log(f"Extracted {n} frames")

    weights_path = ensure_weights()
    upsampler = build_upsampler(device, FP16, SCALE, weights_path)
    upscale_frames(upsampler, FRAMES_DIR, UPSCALED_DIR, SCALE)

    encode_video(UPSCALED_DIR, fps, OUTPUT_PATH)

    # Upload
    upload_blob(bucket, OUTPUT_PATH, output_blob_name, content_type="video/mp4")
    log(f"[DONE] Wrote gs://{BUCKET}/{output_blob_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(2)