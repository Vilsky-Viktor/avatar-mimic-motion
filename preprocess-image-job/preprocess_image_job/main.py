import os, json, logging, io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from preprocess_image_job.gcs_utils import (
    get_client, read_json, list_images,
    download_image, upload_png, download_folder
)
from preprocess_image_job.detectors import FaceDetector, BodyDetector
from preprocess_image_job.crop_utils import crop_face, crop_body
from preprocess_image_job.rmbg_utils import RMBG2

# --------------------------------------------------------------------
# Env / constants
# --------------------------------------------------------------------
BUCKET = os.environ["JOBS_BUCKET"]
EXEC_ID = os.environ["EXECUTION_ID"]

# LoRA target size (portrait 9:16)
TARGET_W, TARGET_H = 576, 1024

# Neutral background color (RGB triplet from env "NEUTRAL_BG=127,127,127")
def _parse_bg_env() -> Tuple[int, int, int]:
    s = os.getenv("NEUTRAL_BG", "127,127,127")
    try:
        r, g, b = (int(x.strip()) for x in s.split(","))
        r = int(np.clip(r, 0, 255)); g = int(np.clip(g, 0, 255)); b = int(np.clip(b, 0, 255))
        return (r, g, b)
    except Exception:
        return (127, 127, 127)

NEUTRAL_BG_RGB = _parse_bg_env()

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def upload_png_rgb(bucket, gcs_path: str, rgb: np.ndarray):
    """Upload a 3-channel RGB numpy array as PNG via Pillow (keeps channel order)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"upload_png_rgb expects HxWx3 RGB, got shape={rgb.shape}")
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    bucket.blob(gcs_path).upload_from_file(buf, content_type="image/png")

def composite_on_bg_to_rgb(img: np.ndarray, bg_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Input can be BGRA/RGBA/GRAY/BGR.
    Returns HxWx3 RGB with alpha composited over bg_rgb.
    """
    if img is None:
        return img
    # 4-channel: treat as BGRA (most OpenCV/RMBG outputs)
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = (a.astype(np.float32) / 255.0)[..., None]  # HxWx1
        fg_rgb = cv2.merge([r, g, b]).astype(np.float32)   # BGRA -> RGB
        bg = np.empty_like(fg_rgb, dtype=np.float32); bg[:] = bg_rgb
        out = fg_rgb * alpha + bg * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    # 3-channel BGR -> RGB
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # GRAY -> RGB
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def letterbox_to_aspect_rgb(img_rgb: np.ndarray, tw: int, th: int,
                            bg_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize preserving aspect, then pad to (tw, th) using bg_rgb.
    Works on RGB arrays. Returns RGB.
    """
    h, w = img_rgb.shape[:2]
    if w == 0 or h == 0:
        return img_rgb

    target_aspect = tw / th
    src_aspect = w / h

    if abs(src_aspect - target_aspect) < 1e-6:
        return cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_CUBIC)

    if src_aspect > target_aspect:
        new_w = tw
        new_h = int(round(new_w / src_aspect))
    else:
        new_h = th
        new_w = int(round(new_h * src_aspect))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    top = (th - new_h) // 2
    bottom = th - new_h - top
    left = (tw - new_w) // 2
    right = tw - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=bg_rgb  # tuple is applied per-channel
    )
    return padded

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    bucket = client.bucket(BUCKET)
    manifest = read_json(bucket, f"jobs/{EXEC_ID}/manifest.json")
    aid = manifest["ai_avatar_id"]

    in_prefix = f"ai_avatars/{aid}/identity_images/"
    out_root = f"ai_avatars/{aid}/"
    imgs = list_images(bucket, in_prefix)

    logging.info("downloading RMBG model ...")
    download_folder(bucket, "models/rmbg-2-0/", Path("/models/rmbg2"))

    face_det, body_det = FaceDetector(), BodyDetector()
    rmbg = RMBG2()

    reports = []
    for path in imgs:
        logging.info(f"processing image {path}")
        orig_img_bgr = download_image(bucket, path)   # BGR (H,W,3)
        name = Path(path).stem

        used, reasons = False, []

        # --- Remove background once (keep alpha artifact for audit) ---
        nobg_bgra = rmbg.remove_bg(orig_img_bgr)  # often BGRA
        upload_png(bucket, f"{out_root}no_background/{name}.png", nobg_bgra)  # audit copy (keeps alpha)

        # --- Face crop ---
        face = face_det.detect(orig_img_bgr)  # detect on original
        if face:
            logging.info("face detected ... cropping")
            fc = crop_face(nobg_bgra, face["box"], face["landmarks"])  # crop from no-bg (likely BGRA)
            if fc is not None:
                # Composite to neutral background in RGB and letterbox
                fc_rgb = composite_on_bg_to_rgb(fc, NEUTRAL_BG_RGB)
                fc_rgb = letterbox_to_aspect_rgb(fc_rgb, TARGET_W, TARGET_H, NEUTRAL_BG_RGB)
                upload_png_rgb(bucket, f"{out_root}face_crops/{name}.png", fc_rgb)  # <-- RGB on disk
                used = True
            else:
                reasons.append("face_crop_failed")
        else:
            reasons.append("face_not_detected")

        # --- Body crop ---
        body = body_det.detect(orig_img_bgr)  # detect on original
        if body:
            logging.info("body detected ... cropping")
            bc = crop_body(nobg_bgra, body["box"], 0.08)  # crop from no-bg (likely BGRA)
            if bc is not None:
                bc_rgb = composite_on_bg_to_rgb(bc, NEUTRAL_BG_RGB)
                bc_rgb = letterbox_to_aspect_rgb(bc_rgb, TARGET_W, TARGET_H, NEUTRAL_BG_RGB)
                upload_png_rgb(bucket, f"{out_root}body_crops/{name}.png", bc_rgb)  # <-- RGB on disk
                used = True
            else:
                reasons.append("body_crop_failed")
        else:
            reasons.append("body_not_detected")

        reports.append({"img": path, "used": used, "reasons": reasons})

    # --- Metadata ---
    logging.info("composing crop report")
    meta = {"execution_id": EXEC_ID, "ai_avatar_id": aid, "images": reports}
    bucket.blob(f"{out_root}crop_report.json").upload_from_string(
        json.dumps(meta, indent=2), "application/json"
    )
    logging.info("[DONE]")

if __name__ == "__main__":
    main()