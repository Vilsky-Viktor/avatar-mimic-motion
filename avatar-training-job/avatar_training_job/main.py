# main.py
import os
import random
from pathlib import Path
import math
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as ckpt

import numpy as np
import cv2
from PIL import Image
import imageio.v3 as iio
import safetensors.torch as st
from tqdm.auto import tqdm

# --- Compatibility shim for huggingface_hub >= 0.20 ---
try:
    import huggingface_hub as _hfh
    import huggingface_hub.file_download as _hffd
    if not hasattr(_hfh, "cached_download"):
        _hfh.cached_download = _hffd.hf_hub_download
except Exception:
    pass

# ---- Diffusers 0.24.x imports ----
from diffusers import StableVideoDiffusionPipeline
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

from accelerate import Accelerator

from avatar_training_job.arcface_utils import ArcFaceID
from avatar_training_job.dataset import SVDDataset


# =============================================================================
# GCS utility stubs (safe local execution if not in the target job environment)
# =============================================================================
try:
    from avatar_training_job.gcs_utils import (
        bucket as gcs_bucket,
        ensure_local_dataset,
        upload_file,
        download_folder,
    )
except ImportError:
    class GCSUtilsStub:
        def bucket(self, b): return self
        def ensure_local_dataset(self, b, remote, local, extensions):
            local.mkdir(parents=True, exist_ok=True)
            (local / "identity.png").touch()
            for i in range(10):
                (local / f"body_{i}.png").touch()
            return [local / f"body_{i}.png" for i in range(10)]
        def upload_file(self, b, remote, local, content_type):
            print(f"STUB: Uploaded {local.name} -> {remote} ({content_type})")
        def download_folder(self, b, remote, local):
            print("STUB: Model Downloaded (no-op in stub).")

    gcs_bucket = lambda b: GCSUtilsStub().bucket(b)
    ensure_local_dataset = GCSUtilsStub().ensure_local_dataset
    upload_file = GCSUtilsStub().upload_file
    download_folder = GCSUtilsStub().download_folder


# =============================================================================
# ENV / Paths
# =============================================================================
BUCKET = os.getenv("JOBS_BUCKET", "your-gcs-bucket-name")
AI_AVATAR_ID = os.getenv("AI_AVATAR_ID", "default-avatar-id")
TARGET_MODEL_NAME = os.getenv("TARGET_MODEL_NAME", "default-lora-model")

BASE_MODEL_PREFIX = "models/mimic_motion/stable-video-diffusion-img2vid-xt-1-1"
OUT_PREFIX = f"ai_avatars/{AI_AVATAR_ID}/models/{TARGET_MODEL_NAME}"

LOCAL_DATA = Path("/data")
LOCAL_MODELS = Path("/models")
LOCAL_OUT = Path("/output")
for d in (LOCAL_DATA, LOCAL_MODELS, LOCAL_OUT):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Hyperparams
# =============================================================================
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "1"))
ACC_STEPS     = int(os.getenv("ACC_STEPS", "8"))         # was 4
EPOCHS        = int(os.getenv("EPOCHS", "50"))
LR            = float(os.getenv("LR", "3e-5"))           # was 4e-5
WD            = float(os.getenv("WD", "0.01"))           # was 0.0
WARMUP_STEPS  = int(os.getenv("WARMUP_STEPS", "500"))
RESTARTS      = int(os.getenv("RESTARTS", "1"))
MAX_STEPS     = int(os.getenv("MAX_STEPS", "0"))

NUM_FRAMES             = int(os.getenv("NUM_FRAMES", "14"))
ARC_FACE_NUM_FRAMES    = int(os.getenv("ARC_FACE_NUM_FRAMES", "3"))
FPS                    = int(os.getenv("FPS", "6"))
TARGET_H               = int(os.getenv("TARGET_H", "1024"))
TARGET_W               = int(os.getenv("TARGET_W", "576"))

LORA_RANK   = int(os.getenv("LORA_RANK", "32"))
LORA_ALPHA  = float(os.getenv("LORA_ALPHA", "32.0"))      # was 64.0

LAMBDA_ID           = float(os.getenv("LAMBDA_ID", "0.25"))
MOTION_BUCKET_ID    = int(os.getenv("MOTION_BUCKET_ID", "120"))
NOISE_AUG_STRENGTH  = float(os.getenv("NOISE_AUG_STRENGTH", "0.012"))

TS_MIN = int(os.getenv("TS_MIN", "200"))
TS_MAX = int(os.getenv("TS_MAX", "900"))

MIXED_PRECISION = os.getenv("MIXED_PRECISION", "bf16")     # "bf16", "fp16", or "no"

USE_EMA   = os.getenv("USE_EMA", "1") == "1"
EMA_DECAY = float(os.getenv("EMA_DECAY", "0.9995"))

# Memory knob for ID path VAE decode
ID_LATENT_DOWNSCALE = int(os.getenv("ID_LATENT_DOWNSCALE", "2"))

print(f"BATCH_SIZE {BATCH_SIZE}")
print(f"ACC_STEPS  {ACC_STEPS}")
print(f"EPOCHS     {EPOCHS}")
print(f"LR         {LR}")
print(f"WD         {WD}")
print(f"WARMUP     {WARMUP_STEPS}")
print(f"LORA (r,alpha)=({LORA_RANK},{LORA_ALPHA})")
print(f"NUM_FRAMES {NUM_FRAMES}  FPS {FPS}  SIZE {TARGET_W}x{TARGET_H}")
print(f"ID λ {LAMBDA_ID}  MOTION_BUCKET_ID {MOTION_BUCKET_ID}  NOISE_AUG {NOISE_AUG_STRENGTH}")
print(f"TS window [{TS_MIN},{TS_MAX})")
print(f"MIXED_PRECISION {MIXED_PRECISION}  USE_EMA {USE_EMA} (decay={EMA_DECAY})")
print(f"ID_LATENT_DOWNSCALE {ID_LATENT_DOWNSCALE}")

# =============================================================================
# Helpers
# =============================================================================

def _mem_str():
    if not torch.cuda.is_available(): return "cpu"
    free, total = torch.cuda.mem_get_info()
    return f"{(total-free)//(1024**2)}/{total//(1024**2)}MB"

def _grad_global_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None: continue
        v = p.grad.detach()
        total += float(v.pow(2).sum().item())
    return math.sqrt(total) if total > 0 else 0.0

class _LinearProbe(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)

class ZeroAddEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, time_out_features: int):
        super().__init__()
        self.linear_1 = _LinearProbe(in_features, time_out_features)
        self.linear_2 = _LinearProbe(time_out_features, time_out_features)
    def forward(self, time_embeds: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (time_embeds.shape[0], self.linear_2.out_features),
            device=time_embeds.device,
            dtype=time_embeds.dtype,
        )

def maybe_neutralize_add_embedding(unet, device, mp_dtype) -> bool:
    try:
        time_dim = int(unet.time_embedding.linear_2.out_features)
        add_dim  = int(getattr(unet.config, "addition_time_embed_dim", 0) or 0)
        lin1 = getattr(getattr(unet, "add_embedding", None), "linear_1", None)
        in_feat = int(lin1.in_features) if lin1 is not None else None

        print(f"[CHK] add_embedding.linear_1.in_features: {in_feat}")
        print(f"[CHK] time_embedding.linear_2.out_features: {time_dim}")
        print(f"[CHK] addition_time_embed_dim (cfg): {add_dim}")
        print(f"[CHK] projection_class_embeddings_input_dim (cfg): {getattr(unet.config, 'projection_class_embeddings_input_dim', None)}")

        expected_in = time_dim + add_dim
        needs_patch = in_feat is not None and in_feat != expected_in

        if needs_patch:
            proj_in = int(getattr(unet.config, "projection_class_embeddings_input_dim", 768))
            print(f"[FIX] add_embedding mismatch: expects {in_feat}, runtime feeds {expected_in}. Neutralizing.")
            print(f"[FIX] Installing ZeroAddEmbedding(in_features={proj_in}, time_dim={time_dim})")
            unet.add_embedding = ZeroAddEmbedding(in_features=proj_in, time_out_features=time_dim).to(
                device=device, dtype=mp_dtype
            )
            unet.config.addition_time_embed_dim = 0
            # Do not touch unet.addition_time_embed_dim directly (avoids FutureWarning).
            for p in unet.add_embedding.parameters():
                p.requires_grad = False
            print("[FIX] add_embedding neutralized; addition_time_embed_dim set to 0 in config.")
            return True
        else:
            print("[OK] add_embedding dimensions are consistent.")
            return False
    except Exception as e:
        print(f"[WARN] add_embedding check failed (continuing unpatched): {e}")
        return False

def load_and_prepare_bgr_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

# ---------- Robust LoRA injection (no guessed names, no get_attn_processors) ----------
def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    """
    Traverse `root` following a dotted path. Supports numeric tokens for ModuleList indices.
    Example key: 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
    We pass everything up to '.processor' into this function.
    """
    cur = root
    for tok in dotted.split("."):
        if tok == "":
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur

def _infer_hidden_and_cross(attn_mod: nn.Module) -> tuple[int, Optional[int]]:
    """
    Infer hidden_size and cross_attention_dim from the attention module:
      hidden_size := attn_mod.to_q.in_features
      cross_attention_dim := None if to_k.in_features == hidden_size (self-attn)
                              else to_k.in_features (cross-attn)
    """
    to_q = getattr(attn_mod, "to_q", None)
    to_k = getattr(attn_mod, "to_k", None)
    if to_q is None or to_k is None or not hasattr(to_q, "in_features") or not hasattr(to_k, "in_features"):
        raise RuntimeError(f"Cannot infer dims from attention module {type(attn_mod).__name__}")
    hidden_size = int(to_q.in_features)
    k_in = int(to_k.in_features)
    cross_dim = None if k_in == hidden_size else k_in
    return hidden_size, cross_dim

def inject_lora_safe(unet, r: int, alpha: float):
    """
    Robust LoRA injection for UNetSpatioTemporalConditionModel on diffusers 0.24.x.
    Iterates over `unet.attn_processors`, resolves each owning attention module,
    infers dims, and replaces the processor with a LoRA processor.
    """
    use_sdpa = hasattr(F, "scaled_dot_product_attention")
    LORA_CLS = LoRAAttnProcessor2_0 if use_sdpa else LoRAAttnProcessor

    base = getattr(unet, "attn_processors", None)
    if not isinstance(base, dict) or len(base) == 0:
        print("[LoRA] ERROR: no attention processors mapping found on UNet; cannot inject.")
        return unet

    new_map = {}
    injected = 0
    for full_name, old_proc in base.items():
        if not (isinstance(full_name, str) and full_name.endswith(".processor")):
            # Keep any unexpected entries as-is
            new_map[full_name] = old_proc
            continue

        owner_path = full_name[: -len(".processor")]
        try:
            attn_mod = _resolve_module(unet, owner_path)
            hidden_size, cross_dim = _infer_hidden_and_cross(attn_mod)
            new_map[full_name] = LORA_CLS(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                rank=r,
                network_alpha=alpha,
            )
            injected += 1
        except Exception as e:
            new_map[full_name] = old_proc
            print(f"[LoRA] WARN: skipped {full_name}: {e}")

    if injected == 0:
        print("[LoRA] ERROR: injected 0 processors; training would be a no-op.")
    else:
        unet.set_attn_processor(new_map)
        print(f"[LoRA] Injected LoRA into {injected} attention processors.")
        sample_keys = [k for k, v in new_map.items() if isinstance(v, (LoRAAttnProcessor, LoRAAttnProcessor2_0))][:6]
        if sample_keys:
            print("[LoRA] sample injected keys:", ", ".join(sample_keys))
    return unet

def save_lora_weights(unet, out_path: Path):
    procs = getattr(unet, "attn_processors", None)
    if not isinstance(procs, dict) or len(procs) == 0:
        procs = {}
    state = {}
    for name, proc in procs.items():
        if hasattr(proc, "state_dict"):
            sd = proc.state_dict()
            for p_name, p in sd.items():
                state[f"{name}.{p_name}"] = p.detach().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.save_file(state, str(out_path))
    print(f"[LoRA] Saved {len(state)} tensors → {out_path}")

def build_added_time_ids(b: int, latents_hw, fps, motion_bucket_id, noise_aug_strength, dtype, device):
    h, w = latents_hw
    vec = torch.tensor(
        [float(fps), float(motion_bucket_id), float(noise_aug_strength), float(h), float(w), 0.0],
        dtype=dtype, device=device,
    )
    return vec.unsqueeze(0).repeat(b, 1)

def fix_group_norm_channels(unet):
    try:
        down_blocks = getattr(unet, "down_blocks", None)
        if not down_blocks:
            print("[GROUPNORM-FIX] Skipped (no down_blocks).")
            return
        first = down_blocks[0]
        if not hasattr(first, "resnets"):
            print("[GROUPNORM-FIX] Skipped (first down_block has no resnets).")
            return
        device = next(unet.parameters()).device
        patched = 0
        for i, resnet in enumerate(first.resnets):
            inner = resnet
            if hasattr(resnet, "spatial_res_block") and hasattr(resnet.spatial_res_block, "conv1"):
                inner = resnet.spatial_res_block
                expected = inner.conv1.out_channels
            elif hasattr(resnet, "conv1"):
                expected = resnet.conv1.out_channels
            else:
                continue
            if hasattr(inner, "norm1") and isinstance(inner.norm1, nn.GroupNorm):
                old = inner.norm1
                if old.num_channels != expected:
                    new = nn.GroupNorm(
                        num_groups=min(old.num_groups, expected),
                        num_channels=expected,
                        eps=old.eps,
                        affine=True,
                    )
                    if new.weight.shape == old.weight.shape:
                        new.weight.data.copy_(old.weight.data)
                        new.bias.data.copy_(old.bias.data)
                    inner.norm1 = new.to(device)
                    print(f"[GROUPNORM-FIX] Patched norm1 in down_blocks[0].resnets[{i}] to {expected} ch.")
                    patched += 1
            if hasattr(inner, "norm2") and isinstance(inner.norm2, nn.GroupNorm):
                old2 = inner.norm2
                if old2.num_channels != expected:
                    new2 = nn.GroupNorm(
                        num_groups=min(old2.num_groups, expected),
                        num_channels=expected,
                        eps=old2.eps,
                        affine=True,
                    )
                    if new2.weight.shape == old2.weight.shape:
                        new2.weight.data.copy_(old2.weight.data)
                        new2.bias.data.copy_(old2.bias.data)
                    inner.norm2 = new2.to(device)
                    print(f"[GROUPNORM-FIX] Patched norm2 in down_blocks[0].resnets[{i}] to {expected} ch.")
                    patched += 1
        if patched == 0:
            print("[GROUPNORM-FIX] No changes applied (already consistent).")
    except Exception as e:
        print(f"[GROUPNORM-FIX] Skipped due to exception: {e}")

def collect_lora_parameters(unet):
    params = []
    procs = getattr(unet, "attn_processors", None)
    if not isinstance(procs, dict) or len(procs) == 0:
        return params
    for proc in procs.values():
        if isinstance(proc, (LoRAAttnProcessor, LoRAAttnProcessor2_0)):
            params += list(proc.parameters())
    return params

def _worker_init_fn(worker_id):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)

# --- EMA utils ---
def build_ema(params, device):
    return [p.detach().clone().to(device=device, dtype=torch.float32) for p in params]

@torch.no_grad()
def ema_update(shadow, params, decay: float):
    for s, p in zip(shadow, params):
        s.mul_(decay).add_(p.detach().to(dtype=torch.float32), alpha=1.0 - decay)

@torch.no_grad()
def swap_in_ema(params, shadow):
    backup = [p.detach().clone() for p in params]
    for p, s in zip(params, shadow):
        p.data.copy_(s.to(dtype=p.dtype, device=p.device))
    return backup

@torch.no_grad()
def restore_from_backup(params, backup):
    for p, b in zip(params, backup):
        p.data.copy_(b.to(dtype=p.dtype, device=p.device))


# =============================================================================
# Main
# =============================================================================
def main():
    # Repro + perf knobs
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("[ENV] torch", torch.__version__)
    import diffusers as _df
    print("[ENV] diffusers", _df.__version__)  # expect 0.24.x

    bkt = gcs_bucket(BUCKET)

    # 1) Dataset
    face_paths = ensure_local_dataset(bkt, f"ai_avatars/{AI_AVATAR_ID}/face_crops", LOCAL_DATA / "faces", (".png", ".jpg", ".jpeg"))
    body_paths = ensure_local_dataset(bkt, f"ai_avatars/{AI_AVATAR_ID}/body_crops", LOCAL_DATA / "bodies", (".png", ".jpg", ".jpeg"))

    all_paths = body_paths + face_paths
    ds = SVDDataset(all_paths)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if 2 > 0 else False,
        worker_init_fn=_worker_init_fn
    )

    # 2) Model
    local_model_dir = LOCAL_MODELS / "svd_1_1"
    if local_model_dir.exists():
        import shutil; shutil.rmtree(local_model_dir)
    local_model_dir.mkdir(parents=True, exist_ok=True)
    download_folder(bkt, BASE_MODEL_PREFIX, local_model_dir)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(local_model_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # memory knobs
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"[WARN] xFormers not enabled: {e}")
    try:
        pipe.unet.enable_gradient_checkpointing()
    except Exception as e:
        print(f"[WARN] gradient checkpointing not enabled: {e}")

    # 3) Accelerator & dtype
    accelerator = Accelerator(
        gradient_accumulation_steps=ACC_STEPS,
        mixed_precision=MIXED_PRECISION if torch.cuda.is_available() else "no",
    )
    device = accelerator.device
    mp_dtype = (
        torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.float32
    )

    # Move submodules
    pipe.vae = pipe.vae.to(device, dtype=mp_dtype)

    # 4) ArcFace anchor (Torch path)
    arc = ArcFaceID()
    anchor_emb = torch.zeros(512, dtype=torch.float32, device=device)
    if arc.is_ready:
        anchor_img = load_and_prepare_bgr_image(LOCAL_DATA / "faces" / "identity.png")
        if anchor_img is not None:
            anc_rgb = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
            anc_rgb = torch.from_numpy(anc_rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
            anc_rgb = anc_rgb.unsqueeze(0).to(device)                              # [1,3,H,W]
            with torch.no_grad():
                anchor_emb = arc.embed_torch(anc_rgb, do_crop=True)[0]  # L2-normalized [512]
        else:
            print("WARNING: identity.png not found or empty.")
    else:
        print("WARNING: ArcFace torch embedder not available; identity loss will be disabled.")

    # 5) Patch add_embedding *after* accelerator/dtype are known
    maybe_neutralize_add_embedding(pipe.unet, device, mp_dtype)

    # Sanity prints
    u = pipe.unet
    try:
        print("[CHK] add_embedding is callable:", callable(getattr(u, "add_embedding", None)))
        print("[CHK] time_embedding.linear_2.out_features:", u.time_embedding.linear_2.out_features)
        print("[CHK] addition_time_embed_dim (cfg):", getattr(u.config, "addition_time_embed_dim", None))
        print("[CHK] projection_class_embeddings_input_dim (cfg):", getattr(u.config, "projection_class_embeddings_input_dim", None))
    except Exception as e:
        print("[CHK] UNet embedding prints skipped:", e)

    # 6) LoRA + GN fix
    inject_lora_safe(pipe.unet, r=LORA_RANK, alpha=LORA_ALPHA)
    fix_group_norm_channels(pipe.unet)

    # Sanity check: ensure we actually injected LoRA
    procs_after = getattr(pipe.unet, "attn_processors", {}) or {}
    n_lora = sum(isinstance(p, (LoRAAttnProcessor, LoRAAttnProcessor2_0)) for p in procs_after.values())
    print(f"[LoRA] processors total={len(procs_after)}  lora_processors={n_lora}")
    assert len(procs_after) > 0 and n_lora > 0, "LoRA injection failed (0 LoRA processors)."

    # 7) Trainables, optimizer, scheduler
    trainable_params = collect_lora_parameters(pipe.unet)
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were found!")

    opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    total_train_steps = MAX_STEPS if MAX_STEPS > 0 else (math.ceil(len(dl) / ACC_STEPS) * EPOCHS)
    lr_sched = get_cosine_with_hard_restarts_schedule_with_warmup(opt, WARMUP_STEPS, total_train_steps, RESTARTS)

    opt, dl, lr_sched = accelerator.prepare(opt, dl, lr_sched)
    pipe.unet = pipe.unet.to(device)
    pipe.unet.train()

    # 8) EMA
    ema_shadow = build_ema(trainable_params, device=device) if USE_EMA else None

    # Cross-attn context (zeros)
    cross_dim = int(getattr(pipe.unet.config, "cross_attention_dim", 0))
    if cross_dim <= 0:
        raise RuntimeError(f"Invalid cross_attention_dim={cross_dim} (must be > 0 for SVD).")

    global_step = 0

    for epoch in range(EPOCHS):
        dl_iter = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(dl_iter):
            if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                break

            DEBUG = os.getenv("DEBUG_SVD", "1") == "1"

            def _dprint(*args):
                if DEBUG and (not hasattr(accelerator, "is_main_process") or accelerator.is_main_process):
                    print(*args, flush=True)

            with accelerator.accumulate(pipe.unet):
                try:
                    # -------------------- Prep inputs --------------------
                    bodies = batch["body_bgr"]  # uint8 [B,H,W,3] BGR
                    _dprint(f"[DBG] bodies: shape={tuple(bodies.shape)}, dtype={bodies.dtype}, "
                            f"min={int(bodies.min()) if bodies.numel() else 'NA'} max={int(bodies.max()) if bodies.numel() else 'NA'}")

                    # BGR -> RGB, CHW, [0,1], move to device/dtype
                    cond_rgb = bodies[..., [2, 1, 0]].to(device).float().div_(255.0)
                    cond_rgb = cond_rgb.permute(0, 3, 1, 2).to(dtype=mp_dtype)
                    _dprint(f"[DBG] cond_rgb: shape={tuple(cond_rgb.shape)}, dtype={cond_rgb.dtype}, device={cond_rgb.device}")

                    # -------------------- Latents & noise --------------------
                    with torch.no_grad():
                        cond_latent = pipe.vae.encode(cond_rgb * 2 - 1).latent_dist.sample()
                        cond_latent = cond_latent * pipe.vae.config.scaling_factor  # [B,4,H/8,W/8]
                        b, c, lh, lw = cond_latent.shape
                        _dprint(f"[DBG] cond_latent: shape={tuple(cond_latent.shape)}, scaling_factor={pipe.vae.config.scaling_factor}")

                        # Timesteps: one per sequence, restricted window
                        base_ts = pipe.scheduler.timesteps.to(cond_latent.device)
                        lo = max(0, min(TS_MIN, base_ts.numel()-1))
                        hi = max(lo+1, min(TS_MAX, base_ts.numel()))
                        t_slice = base_ts[lo:hi]
                        t_seq = t_slice[torch.randint(0, t_slice.numel(), (b,), device=cond_latent.device)]  # [B]
                        t_per_frame = t_seq.repeat_interleave(NUM_FRAMES)  # [B*F]
                        _dprint(f"[DBG] timesteps: base_len={base_ts.numel()}, range=[{lo},{hi}), "
                                f"t_seq.shape={tuple(t_seq.shape)}, t_per_frame.shape={tuple(t_per_frame.shape)}, "
                                f"t_seq[:min(4,b)]={t_seq[:min(4,b)].tolist()}")

                        # Replicate latents across frames
                        target_latents = (
                            cond_latent.unsqueeze(1)
                            .expand(b, NUM_FRAMES, c, lh, lw)
                            .reshape(b * NUM_FRAMES, c, lh, lw)
                        )  # [B*F,4,*,*]

                        noise = torch.randn_like(target_latents)
                        noisy_latents_flat = pipe.scheduler.add_noise(target_latents, noise, t_per_frame)  # [B*F,4,*,*]

                        # Concatenate conditioning latents -> in_channels=8
                        cond_latent_rep = (
                            cond_latent.unsqueeze(1)
                            .expand(b, NUM_FRAMES, c, lh, lw)
                            .reshape_as(target_latents)
                        )
                        noisy_latents_flat = torch.cat([noisy_latents_flat, cond_latent_rep], dim=1)  # [B*F,8,*,*]
                        _dprint(f"[DBG] target_latents={tuple(target_latents.shape)}, noise={tuple(noise.shape)}, "
                                f"noisy_latents_flat={tuple(noisy_latents_flat.shape)}")

                    # Reshape to 3D UNet input [B,F,C,H,W]
                    _, c_pad, h, w = noisy_latents_flat.shape
                    noisy_latents = noisy_latents_flat.view(b, NUM_FRAMES, c_pad, h, w)
                    _dprint(f"[DBG] noisy_latents (3D): shape={tuple(noisy_latents.shape)}")

                    # Cross-attn encoder context (zeros)
                    encoder_ctx = torch.zeros(b, 1, cross_dim, dtype=noisy_latents.dtype, device=noisy_latents.device)
                    _dprint(f"[DBG] encoder_ctx: shape={tuple(encoder_ctx.shape)}, dtype={encoder_ctx.dtype}")

                    # Required: added_time_ids [B,6]
                    added_time_ids = build_added_time_ids(
                        b, (h, w), FPS, MOTION_BUCKET_ID, NOISE_AUG_STRENGTH,
                        noisy_latents.dtype, noisy_latents.device
                    )
                    _dprint(f"[DBG] added_time_ids: shape={tuple(added_time_ids.shape)}, sample[0]={added_time_ids[0].tolist()}")

                    # -------------------- UNet forward --------------------
                    with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=(accelerator.mixed_precision != "no")):
                        out = pipe.unet(
                            noisy_latents,                 # [B,F,8,H/8,W/8]
                            t_seq,                         # [B]
                            encoder_hidden_states=encoder_ctx,
                            added_time_ids=added_time_ids, # [B,6]
                        )
                    _dprint(f"[DBG] unet out.sample: shape={tuple(out.sample.shape)}, dtype={out.sample.dtype}")

                except Exception as e:
                    _dprint(f"[ERR] Exception in forward: {repr(e)}")
                    try:
                        _dprint(f"[ERR] unet.config.addition_time_embed_dim={getattr(pipe.unet.config,'addition_time_embed_dim',None)}")
                        _dprint(f"[ERR] projection_class_embeddings_input_dim={getattr(pipe.unet.config,'projection_class_embeddings_input_dim',None)}")
                    except Exception:
                        pass
                    try:
                        _dprint(f"[ERR] cross_attention_dim={getattr(pipe.unet.config,'cross_attention_dim',None)}")
                    except Exception:
                        pass
                    raise

                # -------------------- Losses --------------------
                model_pred = out.sample.flatten(0, 1)   # [B*F, 8, H/8, W/8]
                model_pred_noise = model_pred[:, :4]    # predicted noise
                loss_recon = F.mse_loss(model_pred_noise.float(), noise.float())

                # Identity loss — DIFFERENTIABLE with Euler sigmas
                total_id_loss = torch.tensor(0.0, device=device)
                faces_used = 0

                if arc.is_ready and LAMBDA_ID > 0:
                    sel_indices = []
                    for i in range(b):
                        js = random.sample(range(NUM_FRAMES), min(ARC_FACE_NUM_FRAMES, NUM_FRAMES))
                        sel_indices.extend([i * NUM_FRAMES + j for j in js])

                    if len(sel_indices) > 0:
                        noisy4_sel = noisy_latents_flat[sel_indices, :4]     # [K,4,h,w]
                        pred4_sel  = model_pred_noise[sel_indices]           # [K,4,h,w]
                        t_sel      = t_per_frame[sel_indices]                # [K]

                        timesteps = pipe.scheduler.timesteps.to(noisy4_sel.device)   # [N]
                        sigmas    = pipe.scheduler.sigmas.to(noisy4_sel.device)      # [N]

                        if t_sel.ndim == 0:
                            t_sel = t_sel.view(1)
                        match = (t_sel.view(-1, 1) == timesteps.view(1, -1))         # [K,N]
                        has_match = match.any(dim=1)
                        idx_exact = match.float().argmax(dim=1)
                        nearest = (timesteps.float().view(1, -1) - t_sel.float().view(-1, 1)).abs().argmin(dim=1)
                        idx = torch.where(has_match, idx_exact, nearest)             # [K]
                        sigma_t = sigmas[idx].view(-1, 1, 1, 1)                      # [K,1,1,1]

                        x0_latent = noisy4_sel - sigma_t * pred4_sel                 # [K,4,h,w]

                        # ---- VAE decode for ID path (memory-safe) ----
                        vae_dtype = next(pipe.vae.parameters()).dtype
                        latents_dec = (x0_latent / pipe.vae.config.scaling_factor).to(dtype=vae_dtype, device=device)

                        if ID_LATENT_DOWNSCALE > 1:
                            scale = 1.0 / float(ID_LATENT_DOWNSCALE)
                            latents_dec = F.interpolate(latents_dec, scale_factor=scale, mode="bilinear", align_corners=False)

                        def _vae_decode(inp):
                            # No image_only_indicator kwarg for this diffusers version
                            return pipe.vae.decode(inp, num_frames=1).sample

                        with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=(accelerator.mixed_precision != "no")):
                            rec = ckpt(_vae_decode, latents_dec, use_reentrant=False)

                        rec_img01 = (rec.clamp(-1, 1) * 0.5 + 0.5).to(torch.float32)  # [K,3,H,W] fp32 for embedder

                        # Run Torch face embedder (returns L2-normalized embeddings)
                        with torch.autocast(device_type=device.type, enabled=False):
                            emb_k = arc.embed_torch(rec_img01, do_crop=True)         # [K,512], normalized
                            id_sim = F.cosine_similarity(
                                emb_k, anchor_emb.unsqueeze(0).expand_as(emb_k), dim=1
                            )  # [K]
                            total_id_loss = (1.0 - id_sim).mean()

                        faces_used = int(emb_k.shape[0])

                    id_loss_avg = total_id_loss * LAMBDA_ID
                else:
                    id_loss_avg = torch.tensor(0.0, device=device)

                loss = loss_recon + id_loss_avg
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_before = _grad_global_norm(trainable_params)
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                    grad_after = _grad_global_norm(trainable_params)

                    opt.step()
                    lr_sched.step()

                    if USE_EMA and ema_shadow is not None:
                        with torch.no_grad():
                            ema_update(ema_shadow, trainable_params, EMA_DECAY)

                    global_step += 1

                    if accelerator.is_main_process:
                        with torch.no_grad():
                            lr_val  = opt.param_groups[0]["lr"]
                            temp_mid = float(t_seq.mean().item()) if isinstance(t_seq, torch.Tensor) else 0.0
                            print(
                                f"[STEP] e={epoch+1} s={global_step} lr={lr_val:.2e} "
                                f"recon={float(loss_recon.item()):.6f} id={float(total_id_loss):.6f} "
                                f"id_w={float(id_loss_avg):.3f} temp={temp_mid:.6f} total={float(loss.item()):.6f} "
                                f"grad_raw={grad_before:.3f} grad={grad_after:.3f} mem={_mem_str()} id_frames={faces_used}"
                            )

                    opt.zero_grad()

                    if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                        break
                else:
                    pass

        # 9) Save checkpoint per epoch (both regular + EMA)
        if accelerator.is_main_process:
            unet_unwrapped = accelerator.unwrap_model(pipe.unet)

            ckpt_path = LOCAL_OUT / f"{TARGET_MODEL_NAME}_epoch{epoch+1}.safetensors"
            save_lora_weights(unet_unwrapped, ckpt_path)
            upload_file(bkt, f"{OUT_PREFIX}/{ckpt_path.name}", ckpt_path, content_type="application/octet-stream")

            if USE_EMA and ema_shadow is not None:
                backup = swap_in_ema(trainable_params, ema_shadow)
                ckpt_ema = LOCAL_OUT / f"{TARGET_MODEL_NAME}_epoch{epoch+1}.ema.safetensors"
                save_lora_weights(unet_unwrapped, ckpt_ema)
                upload_file(bkt, f"{OUT_PREFIX}/{ckpt_ema.name}", ckpt_ema, content_type="application/octet-stream")
                restore_from_backup(trainable_params, backup)

        if MAX_STEPS > 0 and global_step >= MAX_STEPS:
            break

    # 10) Final save & Test generation
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(pipe.unet)

        final_w = LOCAL_OUT / f"{TARGET_MODEL_NAME}.safetensors"
        save_lora_weights(unet_unwrapped, final_w)
        upload_file(bkt, f"{OUT_PREFIX}/{final_w.name}", final_w, content_type="application/octet-stream")

        if USE_EMA and ema_shadow is not None:
            backup = swap_in_ema(trainable_params, ema_shadow)
            final_ema = LOCAL_OUT / f"{TARGET_MODEL_NAME}.ema.safetensors"
            save_lora_weights(unet_unwrapped, final_ema)
            upload_file(bkt, f"{OUT_PREFIX}/{final_ema.name}", final_ema, content_type="application/octet-stream")
            restore_from_backup(trainable_params, backup)

        # --- quick smoke test video ---
        test_cond = load_and_prepare_bgr_image(LOCAL_DATA / "bodies" / "identity.png")
        if test_cond is None:
            print("WARNING: Skipping test generation due to missing/empty identity.png.")
        else:
            test_rgb = cv2.cvtColor(test_cond, cv2.COLOR_BGR2RGB)
            test_image_pil = Image.fromarray(test_rgb)

            pipe.to(accelerator.device)
            infer_dtype = next(pipe.unet.parameters()).dtype
            pipe.to(torch_dtype=infer_dtype)
            if getattr(pipe, "image_encoder", None) is not None:
                pipe.image_encoder.to(accelerator.device, dtype=infer_dtype).eval()
            if getattr(pipe, "vae", None) is not None:
                pipe.vae.to(accelerator.device, dtype=infer_dtype).eval()
            pipe.unet.eval()

            use_amp = (accelerator.device.type == "cuda" and infer_dtype in (torch.float16, torch.bfloat16))
            amp = torch.autocast("cuda", dtype=infer_dtype) if use_amp else nullcontext()

            gen = torch.Generator(device="cpu").manual_seed(42)

            with torch.inference_mode(), amp:
                result = pipe(
                    image=test_image_pil,
                    num_frames=NUM_FRAMES,
                    decode_chunk_size=4,
                    fps=FPS,
                    height=TARGET_H,
                    width=TARGET_W,
                    generator=gen,
                )
                frames = result.frames[0]

            test_video = LOCAL_OUT / f"{TARGET_MODEL_NAME}.mp4"
            frames_np = [np.array(f) for f in frames]
            iio.imwrite(test_video, frames_np, fps=FPS, codec="h264", quality=8)
            upload_file(bkt, f"{OUT_PREFIX}/{test_video.name}", test_video, content_type="video/mp4")

        print(f"[DONE] Training finished. Final files uploaded to gs://{BUCKET}/{OUT_PREFIX}")


if __name__ == "__main__":
    main()