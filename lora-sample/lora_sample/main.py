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
TARGET_MODEL_FOLDER = os.getenv("TARGET_MODEL_FOLDER", "white-shirt")
TARGET_MODEL_PATH = os.getenv(
    "TARGET_MODEL_PATH",
    f"ai_avatars/{AI_AVATAR_ID}/models/{TARGET_MODEL_FOLDER}/model_name.safetensors",
)

# GCS paths for assets
BASE_MODEL_PREFIX = "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1"
CONDITIONING_IMAGE_PATH_REMOTE = f"ai_avatars/{AI_AVATAR_ID}/body_crops/identity.png"
VIDEO_OUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/models/{TARGET_MODEL_FOLDER}"

# Local paths
LOCAL_BASE_MODEL = Path("/models/svd_1_1")
LOCAL_LORA_MODEL = Path("/models/lora/lora.safetensors")
LOCAL_CONDITIONING_IMAGE = Path("/data/identity.png")
LOCAL_OUT = Path("/output")
for d in (
    LOCAL_BASE_MODEL,
    LOCAL_LORA_MODEL.parent,
    LOCAL_CONDITIONING_IMAGE.parent,
    LOCAL_OUT,
):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Generation Hyperparams (tuned for stability/identity)
# =============================================================================
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "12"))             # ↓ fewer frames = less drift
FPS = int(os.getenv("FPS", "6"))
DECODE_CHUNK_SIZE = int(os.getenv("DECODE_CHUNK_SIZE", "8"))  # ↑ if OOM, set back to 4
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "50"))  # ↑ more detail/stability
SEED = int(os.getenv("SEED", "42"))
LORA_RANK_DEFAULT = int(os.getenv("LORA_RANK", "32"))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "bf16")  # 'fp16' or 'bf16'

# Portrait output (WxH)
TARGET_WIDTH = int(os.getenv("WIDTH", "576"))
TARGET_HEIGHT = int(os.getenv("HEIGHT", "1024"))

# LoRA runtime scale (strength)
LORA_SCALE = float(os.getenv("LORA_SCALE", "0.5"))      # ↓ a bit to avoid overconstrained warping

# SVD motion/identity knobs
MOTION_BUCKET_ID = int(os.getenv("MOTION_BUCKET_ID", "52"))       # 48–96 typical; lower = calmer
NOISE_AUG_STRENGTH = float(os.getenv("NOISE_AUG_STRENGTH", "0.02"))  # 0.02–0.05; lower = more identity

# =============================================================================
# LoRA Helpers
# =============================================================================
def inject_lora_into_unet(unet, rank: int):
    """
    Inject LoRA processors into the UNet attention blocks.
    Must be done before loading the LoRA state_dict (manual fallback path).
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
        attn_procs[name] = LORA_CLS(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank,
        )
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

def apply_lora(pipe: StableVideoDiffusionPipeline, lora_path: Path, scale: float):
    """
    Prefer the built-in LoRA loader (maps keys correctly).
    Fallback: manual injection + state_dict load (strict=False).
    """
    # 1) Try the official loader (preferred)
    try:
        # adapter name keeps default if unspecified
        adapter_name = "default"
        pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[scale])
        print(f"[LoRA] Loaded via pipe.load_lora_weights(adapter='{adapter_name}'), scale={scale}")
        return True
    except Exception as e:
        print(f"[LoRA] Built-in load_lora_weights failed ({e}). Falling back to manual injection...")

    # 2) Manual fallback
    state_dict = load_file(str(lora_path))
    inferred_rank = infer_lora_rank(state_dict, LORA_RANK_DEFAULT)
    if inferred_rank != LORA_RANK_DEFAULT:
        print(f"[LoRA] Inferred rank {inferred_rank} (env {LORA_RANK_DEFAULT}). Using {inferred_rank}.")
    inject_lora_into_unet(pipe.unet, rank=inferred_rank)

    compat = pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"[LoRA] manual load_state_dict: missing={len(compat.missing_keys)} unexpected={len(compat.unexpected_keys)}")

    for _, proc in pipe.unet.attn_processors.items():
        if hasattr(proc, "scale"):
            proc.scale = scale
    print(f"[LoRA] Runtime scale set to {scale}")

    # Quick sanity: try to find at least one lora_up tensor to confirm format
    try:
        some_key = next(k for k in state_dict.keys() if k.endswith("lora_up.weight"))
        print(f"[LoRA] sample '{some_key}' norm={state_dict[some_key].float().norm().item():.4f}")
    except StopIteration:
        print("[LoRA] Warning: no '*lora_up.weight' key found in state dict (naming may differ).")

    return True

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

def _set_sdpa_prefs():
    """Prefer PyTorch SDPA kernels (no xFormers). Works across torch versions."""
    print("Using PyTorch SDPA attention (no xFormers).")
    try:
        # Torch ≥ 2.1: functional style
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception as e:
        try:
            # Older style (attribute methods)
            from torch.backends.cuda import sdp_kernel as _sdp
            _sdp.enable_flash_sdp(True)
            _sdp.enable_mem_efficient_sdp(True)
            _sdp.enable_math_sdp(False)
        except Exception as ee:
            print(f"Could not tweak SDPA kernel prefs: {e if 'e' in locals() else ee}")

# =============================================================================
# Main Generation Script
# =============================================================================
def main():
    print("--- SVD+LoRA Video Generation (SDPA, no xFormers) ---")
    torch.set_grad_enabled(False)

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

    _set_sdpa_prefs()

    # 2) GCS bucket
    bkt = gcs_bucket(BUCKET)

    # 3) Download assets
    print("Downloading assets from GCS...")
    download_folder(bkt, BASE_MODEL_PREFIX, LOCAL_BASE_MODEL)
    download_file(bkt, TARGET_MODEL_PATH, LOCAL_LORA_MODEL)
    download_file(bkt, CONDITIONING_IMAGE_PATH_REMOTE, LOCAL_CONDITIONING_IMAGE)
    print("Downloads complete.")

    # 4) Load base pipeline
    print(f"Loading base SVD pipeline from {LOCAL_BASE_MODEL}...")
    # If your checkpoint dir has an fp16 subfolder, keep variant="fp16" for both fp16/bf16 loads.
    variant = "fp16" if dtype in (torch.float16, torch.bfloat16) else None
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(LOCAL_BASE_MODEL),
        torch_dtype=dtype,   # (FutureWarning in diffusers>=0.25; fine for now)
        variant=variant,
    )

    # Memory tweaks
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # 5) Load LoRA with preferred path (fallback to manual if needed)
    print(f"Loading LoRA from {LOCAL_LORA_MODEL}...")
    apply_lora(pipe, LOCAL_LORA_MODEL, LORA_SCALE)

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

    # Call with new stability knobs; if an older diffusers version complains, retry without them.
    def _call_pipe(**kwargs):
        with torch.inference_mode(), amp_ctx:
            return pipe(**kwargs)

    gen_kwargs = dict(
        image=conditioning_image,
        num_frames=NUM_FRAMES,
        decode_chunk_size=DECODE_CHUNK_SIZE,
        fps=FPS,
        height=TARGET_HEIGHT,
        width=TARGET_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        motion_bucket_id=MOTION_BUCKET_ID,
        noise_aug_strength=NOISE_AUG_STRENGTH,
    )

    try:
        result = _call_pipe(**gen_kwargs)
    except TypeError as e:
        print(f"[Warn] Pipeline args not supported ({e}). Retrying without motion/noise knobs.")
        gen_kwargs.pop("motion_bucket_id", None)
        gen_kwargs.pop("noise_aug_strength", None)
        result = _call_pipe(**gen_kwargs)

    frames = result.frames[0]
    print(f"Generated {len(frames)} frames ({TARGET_WIDTH}x{TARGET_HEIGHT}).")

    # 10) Save MP4
    video_out_path = LOCAL_OUT / "lora_sample.mp4"
    print(f"Saving video to {video_out_path}...")
    frames_np = [np.array(f) if not isinstance(f, np.ndarray) else f for f in frames]
    # Note: some players prefer yuv420p; if playback issues, switch to imageio-ffmpeg writer with pix_fmt
    iio.imwrite(
        video_out_path,
        frames_np,
        fps=FPS,
        codec="h264",
        quality=10,
    )

    # 11) Upload to GCS
    video_out_path_remote = f"{VIDEO_OUT_PREFIX}/{video_out_path.name}"
    print(f"Uploading video to gs://{BUCKET}/{video_out_path_remote}...")
    upload_file(bkt, video_out_path_remote, video_out_path, content_type="video/mp4")

    print("[DONE] Video generation complete. File uploaded to GCS.")

if __name__ == "__main__":
    main()