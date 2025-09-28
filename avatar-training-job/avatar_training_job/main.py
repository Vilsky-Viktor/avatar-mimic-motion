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

from avatar_training_job.arcface_utils import ArcFaceID, cosine_sim
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
# Hyperparams (Pulled from ENV with Defaults)
# =============================================================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
ACC_STEPS = int(os.getenv("ACC_STEPS", "2"))
EPOCHS = int(os.getenv("EPOCHS", "12"))
LR = float(os.getenv("LR", "1e-4"))
WD = float(os.getenv("WD", "0"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "60"))
RESTARTS = int(os.getenv("RESTARTS", "1"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))

# Video generation quality
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "14"))
ARC_FACE_NUM_FRAMES = int(os.getenv("ARC_FACE_NUM_FRAMES", "3"))
FPS = int(os.getenv("FPS", "6"))
TARGET_H = 1024
TARGET_W = 576

# LoRA detail and strength
LORA_RANK = int(os.getenv("LORA_RANK", "16"))
LORA_ALPHA = float(os.getenv("LORA_ALPHA", "32.0"))

# Identity and motion control
LAMBDA_ID = float(os.getenv("LAMBDA_ID", "0.25"))
MOTION_BUCKET_ID = int(os.getenv("MOTION_BUCKET_ID", "127"))
NOISE_AUG_STRENGTH = float(os.getenv("NOISE_AUG_STRENGTH", "0.02"))

# Precision (A100: bf16; otherwise fp16)
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "bf16")  # "bf16", "fp16", or "no"

print(f"BATCH_SIZE {BATCH_SIZE}")
print(f"EPOCHS {EPOCHS}")
print(f"LR {LR}")
print(f"WD {WD}")
print(f"ACC_STEPS {ACC_STEPS}")
print(f"LORA_RANK {LORA_RANK}")
print(f"LORA_ALPHA {LORA_ALPHA}")
print(f"NUM_FRAMES {NUM_FRAMES}")
print(f"WARMUP_STEPS {WARMUP_STEPS}")
print(f"LAMBDA_ID {LAMBDA_ID}")


# =============================================================================
# Helpers
# =============================================================================

class _LinearProbe(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # never used; just here to satisfy nn.Module
        return torch.zeros((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)

class ZeroAddEmbedding(torch.nn.Module):
    """
    A no-op add_embedding that:
    - returns zeros with shape [B, time_dim] so it can be added to time_embeds
    - exposes `.linear_1.in_features` for pipeline introspection
    """
    def __init__(self, in_features: int, time_out_features: int):
        super().__init__()
        self.linear_1 = _LinearProbe(in_features, time_out_features)   # exposes `.in_features`
        self.linear_2 = _LinearProbe(time_out_features, time_out_features)

    def forward(self, time_embeds: torch.Tensor) -> torch.Tensor:
        # match UNet’s expected add-embed output: [B, time_dim]
        return torch.zeros(
            (time_embeds.shape[0], self.linear_2.out_features),
            device=time_embeds.device,
            dtype=time_embeds.dtype,
        )


def maybe_neutralize_add_embedding(unet, accelerator) -> bool:
    """
    Only neutralize add_embedding if the checkpoint's module expects a different
    input size than the runtime will feed (time_dim + addition_time_embed_dim).
    Returns True if patched.
    """
    try:
        time_dim = unet.time_embedding.linear_2.out_features  # usually 1280
        add_dim = int(getattr(unet.config, "addition_time_embed_dim", 0) or 0)
        lin1 = getattr(getattr(unet, "add_embedding", None), "linear_1", None)
        in_feat = lin1.in_features if lin1 is not None else None

        print(f"[CHK] add_embedding.linear_1.in_features: {in_feat}")
        print(f"[CHK] time_embedding.linear_2.out_features: {time_dim}")
        print(f"[CHK] addition_time_embed_dim (cfg): {add_dim}")
        print(f"[CHK] projection_class_embeddings_input_dim (cfg): {getattr(unet.config, 'projection_class_embeddings_input_dim', None)}")

        expected_in = time_dim + add_dim
        needs_patch = in_feat is not None and in_feat != expected_in

        if needs_patch:
            print(f"[FIX] add_embedding input mismatch: expects {in_feat}, runtime feeds {expected_in}. Neutralizing add_embedding.")
            proj_in = int(getattr(u.config, "projection_class_embeddings_input_dim", 768))     # from your UNet config
            time_dim = int(u.time_embedding.linear_2.out_features)                              # typically 1280
            print(f"[FIX] Installing ZeroAddEmbedding: in_features={proj_in}, time_dim={time_dim}")
            unet.add_embedding = ZeroAddEmbedding(in_features=proj_in, time_out_features=time_dim).to(
                device=accelerator.device,
                dtype=(torch.bfloat16 if accelerator.mixed_precision == "bf16"
                    else torch.float16 if accelerator.mixed_precision == "fp16"
                    else torch.float32),
            )
            unet.config.addition_time_embed_dim = 0
            if hasattr(unet, "addition_time_embed_dim"):
                setattr(unet, "addition_time_embed_dim", 0)
            for p in unet.add_embedding.parameters():
                p.requires_grad = False
            print("[FIX] add_embedding neutralized; addition_time_embed_dim set to 0.")
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


def inject_lora_safe(unet, r: int, alpha: float):
    """
    LoRA injection compatible with diffusers 0.24.x:
    Prefer the attn_processors mapping when available.
    """
    use_sdpa = hasattr(F, "scaled_dot_product_attention")
    LORA_CLS = LoRAAttnProcessor2_0 if use_sdpa else LoRAAttnProcessor

    injected = 0
    attn_map = getattr(unet, "attn_processors", None)
    if isinstance(attn_map, dict) and len(attn_map) > 0:
        attn_procs = {}
        for name in attn_map.keys():
            # Deduce hidden size by block
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
            # attn1 is self-attn (no cross), attn2 is cross-attn
            cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            attn_procs[name] = LORA_CLS(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                rank=r,
                network_alpha=alpha,
            )
        if attn_procs:
            unet.set_attn_processor(attn_procs)
            injected = len(attn_procs)

    print(f"[LoRA] Injected LoRA into {injected} attention processors via set_attn_processor().")
    return unet


def save_lora_weights(unet, out_path: Path):
    """
    Save LoRA weights by iterating over attention processors mapping.
    """
    procs = getattr(unet, "attn_processors", None)
    if not isinstance(procs, dict) or len(procs) == 0:
        try:
            procs = unet.get_attn_processors()
        except Exception:
            procs = {}

    state = {}
    for name, proc in procs.items():
        if hasattr(proc, "state_dict"):
            sd = proc.state_dict()
            for p_name, p in sd.items():
                state[f"{name}.{p_name}"] = p.detach().cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True    )
    st.save_file(state, str(out_path))
    print(f"[LoRA] Saved {len(state)} tensors → {out_path}")


def build_added_time_ids(b: int, latents_hw, fps, motion_bucket_id, noise_aug_strength, dtype, device):
    """
    SVD-XT 1.1 requires: [fps, motion_bucket_id, noise_aug_strength, latent_h, latent_w, 0.0]
    """
    h, w = latents_hw
    vec = torch.tensor(
        [float(fps), float(motion_bucket_id), float(noise_aug_strength), float(h), float(w), 0.0],
        dtype=dtype, device=device,
    )
    return vec.unsqueeze(0).repeat(b, 1)


def fix_group_norm_channels(unet):
    """
    Generic and safe GN patcher for the first down block’s resnets.
    Avoids importing internal block classes (keeps 0.24 happy).
    """
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

            # norm1
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
                    print(f"[GROUPNORM-FIX] Applied patch to down_blocks[0].resnets[{i}] norm1 (Channels={expected}).")
                    patched += 1

            # norm2
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
                    print(f"[GROUPNORM-FIX] Applied patch to down_blocks[0].resnets[{i}] norm2 (Channels={expected}).")
                    patched += 1

        if patched == 0:
            print("[GROUPNORM-FIX] No changes applied (already consistent).")

    except Exception as e:
        print(f"[GROUPNORM-FIX] Skipped due to exception: {e}")


def collect_lora_parameters(unet):
    params = []
    procs = getattr(unet, "attn_processors", None)
    if not isinstance(procs, dict) or len(procs) == 0:
        try:
            procs = unet.get_attn_processors()
        except Exception:
            procs = {}
    for proc in procs.values():
        if isinstance(proc, (LoRAAttnProcessor, LoRAAttnProcessor2_0)):
            params += list(proc.parameters())
    return params

def _worker_init_fn(worker_id):
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# =============================================================================
# Main
# =============================================================================
def main():
    # Repro + perf knobs
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        # TF32 can speed up training on Ampere+ without hurting quality here
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("[ENV] torch", torch.__version__)
    import diffusers as _df
    print("[ENV] diffusers", _df.__version__)  # expect 0.24.x

    bkt = gcs_bucket(BUCKET)

    # 1) Dataset
    face_paths = ensure_local_dataset(bkt, f"ai_avatars/{AI_AVATAR_ID}/face_crops", LOCAL_DATA / "faces", (".png", ".jpg", ".jpeg"))
    body_paths = ensure_local_dataset(bkt, f"ai_avatars/{AI_AVATAR_ID}/body_crops", LOCAL_DATA / "bodies", (".png", ".jpg", ".jpeg"))

    ds = SVDDataset(body_paths + face_paths)
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

    # Patch ONLY if needed (shape mismatch)
    maybe_neutralize_add_embedding(pipe.unet, accelerator)

    accelerator = Accelerator(
        gradient_accumulation_steps=ACC_STEPS,
        mixed_precision=MIXED_PRECISION if torch.cuda.is_available() else "no",
    )
    device = accelerator.device

    # dtype to use everywhere
    mp_dtype = (
        torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.float32
    )

    # Move submodules
    pipe.vae = pipe.vae.to(device, dtype=mp_dtype)

    # 3) ArcFace anchor
    arc = ArcFaceID()
    anchor_emb = torch.zeros(512, dtype=torch.float32, device=device)
    if arc.is_ready:
        anchor_img = load_and_prepare_bgr_image(LOCAL_DATA / "faces" / "identity.png")
        if anchor_img is not None:
            with torch.inference_mode():
                emb_tmp = arc.embed_bgr(anchor_img)
            if emb_tmp is not None:
                emb_tmp = emb_tmp.to(device)
                anchor_emb = emb_tmp / (emb_tmp.norm(p=2) + 1e-8)
            else:
                print("WARNING: ArcFace could not detect a face in identity.png.")
        else:
            print("WARNING: identity.png not found or empty.")

    # Sanity: print UNet embedding config
    u = pipe.unet
    try:
        print("[CHK] add_embedding is callable:", callable(getattr(u, "add_embedding", None)))
        print("[CHK] time_embedding.linear_2.out_features:", u.time_embedding.linear_2.out_features)
        print("[CHK] addition_time_embed_dim (cfg):", getattr(u.config, "addition_time_embed_dim", None))
        print("[CHK] projection_class_embeddings_input_dim (cfg):", getattr(u.config, "projection_class_embeddings_input_dim", None))
    except Exception as e:
        print("[CHK] UNet embedding prints skipped:", e)

    # LoRA + GN fix
    inject_lora_safe(pipe.unet, r=LORA_RANK, alpha=LORA_ALPHA)
    fix_group_norm_channels(pipe.unet)

    # Trainables
    trainable_params = collect_lora_parameters(pipe.unet)
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were found!")

    opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    total_train_steps = MAX_STEPS if MAX_STEPS > 0 else (math.ceil(len(dl) / ACC_STEPS) * EPOCHS)
    lr_sched = get_cosine_with_hard_restarts_schedule_with_warmup(opt, WARMUP_STEPS, total_train_steps, RESTARTS)

    opt, dl, lr_sched = accelerator.prepare(opt, dl, lr_sched)
    pipe.unet = pipe.unet.to(device)
    pipe.unet.train()

    # Cross-attn context: zeros (SVD img->vid expects context tensor of size cross_attention_dim)
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

                        # Timesteps: one per sequence
                        base_ts = pipe.scheduler.timesteps.to(cond_latent.device)
                        t_seq = base_ts[torch.randint(0, base_ts.numel(), (b,), device=cond_latent.device)]  # [B]
                        t_per_frame = t_seq.repeat_interleave(NUM_FRAMES)  # [B*F]
                        _dprint(f"[DBG] timesteps: base_len={base_ts.numel()}, t_seq.shape={tuple(t_seq.shape)}, "
                                f"t_per_frame.shape={tuple(t_per_frame.shape)}, t_seq[:min(4,b)]={t_seq[:min(4,b)].tolist()}")

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

                    # -------------------- UNet forward (diffusers 0.24.x) --------------------
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

                # Identity loss (random subset of frames)
                total_id_loss = torch.tensor(0.0, device=device)
                if arc.is_ready and LAMBDA_ID > 0:
                    print('Processing ArcFace on random frames')
                    with torch.no_grad():
                        for i in range(b):
                            frame_indices = random.sample(range(NUM_FRAMES), ARC_FACE_NUM_FRAMES)
                            for j in frame_indices:
                                idx = i * NUM_FRAMES + j
                                target_latent_ij = target_latents[idx].unsqueeze(0)
                                model_pred_ij = model_pred_noise[idx].unsqueeze(0)

                                rec_latent = target_latent_ij - model_pred_ij
                                rec = pipe.vae.decode(rec_latent / pipe.vae.config.scaling_factor, num_frames=1).sample
                                rec_img = (rec.clamp(-1, 1) * 0.5 + 0.5)

                                rec_rgb = (rec_img * 255).to(torch.uint8).permute(0, 2, 3, 1)
                                rec_bgr = rec_rgb[0].cpu().numpy()[:, :, ::-1]

                                emb = arc.embed_bgr(rec_bgr)
                                if emb is not None:
                                    emb = emb.to(device); emb = emb / (emb.norm(p=2) + 1e-8)
                                    id_sim = cosine_sim(emb, anchor_emb)
                                    total_id_loss += (1.0 - id_sim)

                    id_loss_avg = (total_id_loss / max(b * ARC_FACE_NUM_FRAMES, 1)) * LAMBDA_ID
                else:
                    id_loss_avg = torch.tensor(0.0, device=device)

                loss = loss_recon + id_loss_avg
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                opt.step()
                lr_sched.step()
                opt.zero_grad()

            global_step += 1
            if accelerator.is_main_process and global_step % 50 == 0:
                sim_display = 1.0 - (total_id_loss.item() / max(b, 1)) if b > 0 else 0.0
                dl_iter.set_postfix(Recon=float(loss_recon.item()), ID_Sim=float(sim_display), Loss=float(loss.item()))

            if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                break

        # 5) Save checkpoint per epoch
        if accelerator.is_main_process:
            ckpt = LOCAL_OUT / f"{TARGET_MODEL_NAME}_epoch{epoch+1}.safetensors"
            save_lora_weights(accelerator.unwrap_model(pipe.unet), ckpt)
            upload_file(bkt, f"{OUT_PREFIX}/{ckpt.name}", ckpt, content_type="application/octet-stream")

        if MAX_STEPS > 0 and global_step >= MAX_STEPS:
            break

    # 6) Final save & Test generation
    if accelerator.is_main_process:
        final_w = LOCAL_OUT / f"{TARGET_MODEL_NAME}.safetensors"
        save_lora_weights(accelerator.unwrap_model(pipe.unet), final_w)
        upload_file(bkt, f"{OUT_PREFIX}/{final_w.name}", final_w, content_type="application/octet-stream")

        test_cond = load_and_prepare_bgr_image(LOCAL_DATA / "bodies" / "identity.png")
        if test_cond is None:
            print("WARNING: Skipping test generation due to missing/empty identity.png.")
        else:
            # Prep test image (BGR->RGB->PIL)
            test_rgb = cv2.cvtColor(test_cond, cv2.COLOR_BGR2RGB)
            test_image_pil = Image.fromarray(test_rgb)

            # Move ALL pipeline parts to the SAME device + dtype
            pipe.to(accelerator.device)
            infer_dtype = next(pipe.unet.parameters()).dtype  # e.g. torch.float16 or bfloat16
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