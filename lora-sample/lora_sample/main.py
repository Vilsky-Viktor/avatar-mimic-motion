import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import imageio.v3 as iio
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from safetensors.torch import load_file

# --- Compatibility shim for huggingface_hub >= 0.20 ---
try:
    import huggingface_hub as _hfh
    import huggingface_hub.file_download as _hffd
    if not hasattr(_hfh, "cached_download"):
        _hfh.cached_download = _hffd.hf_hub_download
except Exception:
    pass

# Assuming gcs_utils are in a sibling directory or installed package
from lora_sample.gcs_utils import (
    bucket as gcs_bucket,
    upload_file,
    download_folder,
    download_file,
)

# =============================================================================
# ENV / Paths
# =============================================================================
BUCKET = os.getenv("JOBS_BUCKET", "your-gcs-bucket-name")
AI_AVATAR_ID = os.getenv("AI_AVATAR_ID", "default-avatar-id")
TARGET_MODEL_NAME = os.getenv("TARGET_MODEL_NAME", "default-lora-model")

# GCS paths for assets
BASE_MODEL_PREFIX = "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1"
LORA_MODEL_PATH_REMOTE = f"ai_avatars/{AI_AVATAR_ID}/models/{TARGET_MODEL_NAME}/{TARGET_MODEL_NAME}.safetensors"
CONDITIONING_IMAGE_PATH_REMOTE = f"ai_avatars/{AI_AVATAR_ID}/body_crops/identity.png"
VIDEO_OUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/models/{TARGET_MODEL_NAME}"

# Local paths
LOCAL_BASE_MODEL = Path("/models/svd_1_1")
LOCAL_LORA_MODEL = Path(f"/models/lora/{TARGET_MODEL_NAME}.safetensors")
LOCAL_CONDITIONING_IMAGE = Path("/data/identity.png")
LOCAL_OUT = Path("/output")
for d in (LOCAL_BASE_MODEL, LOCAL_LORA_MODEL.parent, LOCAL_CONDITIONING_IMAGE.parent, LOCAL_OUT):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Generation Hyperparams
# =============================================================================
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "14"))
FPS = int(os.getenv("FPS", "6"))
DECODE_CHUNK_SIZE = int(os.getenv("DECODE_CHUNK_SIZE", "4"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "25"))
SEED = int(os.getenv("SEED", "42"))
LORA_RANK_DEFAULT = int(os.getenv("LORA_RANK", "16"))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "bf16")  # 'fp16' or 'bf16'

# Portrait output (WxH)
TARGET_WIDTH = int(os.getenv("WIDTH", "576"))
TARGET_HEIGHT = int(os.getenv("HEIGHT", "1024"))

# LoRA runtime scale (strength)
LORA_SCALE = float(os.getenv("LORA_SCALE", "1.0"))

# =============================================================================
# LoRA Helpers
# =============================================================================
def inject_lora_into_unet(unet, rank: int):
    """
    Inject LoRA processors into the UNet attention blocks.
    Must be done before loading the LoRA state_dict.
    """
    use_sdpa = hasattr(F, "scaled_dot_product_attention")
    LORA_CLS = LoRAAttnProcessor2_0 if use_sdpa else LoRAAttnProcessor

    injected = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[-(block_id + 1)]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        attn_procs[name] = LORA_CLS(hidden_size=hidden_size, cross_attention_dim=cross_dim, rank=rank)
        injected += 1

    unet.set_attn_processor(attn_procs)
    print(f"[LoRA] Injected {injected} attention processors with rank={rank}.")

def infer_lora_rank(state_dict, default_rank: int) -> int:
    """
    Heuristically infer LoRA rank from safetensors by inspecting any '*lora_down.weight'.
    Falls back to default_rank if not found.
    """
    for k, v in state_dict.items():
        if k.endswith("lora_down.weight") or k.endswith("down.weight"):
            try:
                return int(v.shape[0])
            except Exception:
                pass
    return default_rank

def _center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop to target aspect ratio."""
    tw, th = target_w, target_h
    target_aspect = tw / th
    w, h = img.size
    src_aspect = w / h
    if abs(src_aspect - target_aspect) < 1e-6:
        return img
    if src_aspect > target_aspect:
        # too wide: crop width
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        # too tall: crop height
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))

def _enforce_portrait(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Force portrait WxH: center-crop to aspect then resize."""
    img = _center_crop_to_aspect(img, target_w, target_h)
    return img.resize((target_w, target_h), Image.BICUBIC)

# =============================================================================
# Main Generation Script
# =============================================================================
def main():
    print("--- SVD+LoRA Video Generation (SDPA, no xFormers) ---")

    # 1) Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if MIXED_PRECISION == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16 precision on CUDA.")
        else:
            dtype = torch.float16
            print("Using float16 precision on CUDA.")
    else:
        dtype = torch.float32
        print("Using float32 precision on CPU.")

    # Prefer PyTorch SDPA kernels (no xFormers)
    print("Using PyTorch SDPA attention (no xFormers).")
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception as e:
        print(f"Could not tweak SDPA kernel prefs: {e}")

    # 2) GCS bucket
    bkt = gcs_bucket(BUCKET)

    # 3) Download assets
    print("Downloading assets from GCS...")
    download_folder(bkt, BASE_MODEL_PREFIX, LOCAL_BASE_MODEL)
    download_file(bkt, LORA_MODEL_PATH_REMOTE, LOCAL_LORA_MODEL)
    download_file(bkt, CONDITIONING_IMAGE_PATH_REMOTE, LOCAL_CONDITIONING_IMAGE)
    print("Downloads complete.")

    # 4) Load base pipeline
    print(f"Loading base SVD pipeline from {LOCAL_BASE_MODEL}...")
    # If your checkpoint dir has an fp16 subfolder, keep variant="fp16" for both fp16/bf16 loads.
    variant = "fp16" if dtype in (torch.float16, torch.bfloat16) else None
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(LOCAL_BASE_MODEL),
        torch_dtype=dtype,
        variant=variant,
    )
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    # 5) Load LoRA, infer rank, inject processors, load tensors
    print(f"Loading LoRA state dict from {LOCAL_LORA_MODEL}...")
    state_dict = load_file(str(LOCAL_LORA_MODEL))
    inferred_rank = infer_lora_rank(state_dict, LORA_RANK_DEFAULT)
    if inferred_rank != LORA_RANK_DEFAULT:
        print(f"[LoRA] Inferred rank {inferred_rank} (env {LORA_RANK_DEFAULT}). Using {inferred_rank}.")
    inject_lora_into_unet(pipe.unet, rank=inferred_rank)

    print("Loading LoRA weights into UNet (non-strict to allow adapter-only keys)...")
    compat = pipe.unet.load_state_dict(state_dict, strict=False)
    if compat.missing_keys or compat.unexpected_keys:
        print(f"[LoRA] missing={len(compat.missing_keys)} unexpected={len(compat.unexpected_keys)}")
        try:
            print("Missing keys (first 20):", compat.missing_keys[:20])
        except Exception:
            pass

    # Apply runtime LoRA scale
    for name, proc in pipe.unet.attn_processors.items():
        if hasattr(proc, "scale"):
            proc.scale = LORA_SCALE
    print(f"[LoRA] Runtime scale set to {LORA_SCALE}")
    # Quick sanity: print one LoRA tensor norm
    try:
        some_key = next(k for k in state_dict.keys() if k.endswith("lora_up.weight"))
        print(f"[LoRA] sample '{some_key}' norm={state_dict[some_key].float().norm().item():.4f}")
    except StopIteration:
        print("[LoRA] Warning: no '*lora_up.weight' found in state dict.")

    # 6) Move/cast pipeline consistently
    pipe.to(device)
    pipe.to(torch_dtype=dtype)
    pipe.unet.to(dtype=dtype)

    # 7) Conditioning image (force portrait 576x1024 by default)
    print(f"Loading conditioning image from {LOCAL_CONDITIONING_IMAGE}...")
    try:
        image_bgr = cv2.imread(str(LOCAL_CONDITIONING_IMAGE))
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or empty at {LOCAL_CONDITIONING_IMAGE}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        conditioning_image = Image.fromarray(image_rgb)
        conditioning_image = _enforce_portrait(conditioning_image, TARGET_WIDTH, TARGET_HEIGHT)
        print(f"[Input] conditioning_image size={conditioning_image.size} (WxH)")
    except Exception as e:
        print(f"ERROR: Could not load conditioning image. Aborting. Details: {e}")
        return

    # 8) Seeding / generator
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)  # SVD expects CPU generator

    # 9) Generate frames (guard autocast on CPU)
    print(f"Generating video frames with seed {SEED} on {device}/{dtype}...")
    use_amp = (device == "cuda" and dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()
    with torch.inference_mode(), amp_ctx:
        result = pipe(
            image=conditioning_image,
            num_frames=NUM_FRAMES,
            decode_chunk_size=DECODE_CHUNK_SIZE,
            fps=FPS,
            height=TARGET_HEIGHT,
            width=TARGET_WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        )
        frames = result.frames[0]
    print(f"Generated {len(frames)} frames ({TARGET_WIDTH}x{TARGET_HEIGHT}).")

    # 10) Save MP4
    video_out_path = LOCAL_OUT / f"{TARGET_MODEL_NAME}_sample.mp4"
    print(f"Saving video to {video_out_path}...")
    frames_np = [np.array(f) if not isinstance(f, np.ndarray) else f for f in frames]
    iio.imwrite(
        video_out_path,
        frames_np,
        fps=FPS,
        codec="h264",
        quality=10
    )
    # If some players choke on MP4, consider:
    # imageio.get_writer(..., ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart"])

    # 11) Upload to GCS
    video_out_path_remote = f"{VIDEO_OUT_PREFIX}/{video_out_path.name}"
    print(f"Uploading video to gs://{BUCKET}/{video_out_path_remote}...")
    upload_file(bkt, video_out_path_remote, video_out_path, content_type="video/mp4")

    print("[DONE] Video generation complete. File uploaded to GCS.")

if __name__ == "__main__":
    main()