# Vertex AI Custom Job - LoRA training for MimicMotion UNet (Diffusers 0.27.0)
# - Uses MimicMotion UNet forward(sample, timestep, encoder_hidden_states,
#   added_time_ids, pose_latents, image_only_indicator, return_dict)
# - ArcFace identity on 1-frame reconstructions
# - JSON logs per step/epoch (print -> Cloud Logging parses as INFO)
# - Saves epoch & final LoRA adapters to GCS

import os, re, json, math, time, random, inspect, importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2

from google.cloud import storage
from diffusers import StableVideoDiffusionPipeline, DDPMScheduler

# -------------------------
# Logging helpers
# -------------------------
def jlog(event: str, **kv):
    """Emit JSON-ish logs friendly to Cloud Logging (printed as INFO)."""
    def _conv(v):
        try:
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return float(v.detach().cpu().item())
                return f"tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, (Path,)):
                return str(v)
        except Exception:
            pass
        return v
    payload = {"event": event, "timestamp": time.time(), **{k: _conv(v) for k, v in kv.items()}}
    print(json.dumps(payload))

# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    jobs_bucket: str
    avatar_id: str
    local_root: Path = Path("/workspace")
    crop_h: int = int(os.getenv("CROP_H", "1024"))
    crop_w: int = int(os.getenv("CROP_W", "576"))
    epochs: int = int(os.getenv("EPOCHS", "20"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    lr: float = float(os.getenv("LR", "7.5e-5"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.0"))
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "1"))
    mixed_precision: str = os.getenv("MIXED_PRECISION", "bf16")
    seed: int = int(os.getenv("SEED", "42"))
    max_train_steps: Optional[int] = int(os.getenv("MAX_TRAIN_STEPS", "0")) or None
    save_every_n_steps: int = int(os.getenv("SAVE_EVERY_N_STEPS", "200"))
    lora_rank: int = int(os.getenv("LORA_RANK", "64"))
    lora_alpha: float = float(os.getenv("LORA_ALPHA", "64"))
    lora_scale: float = float(os.getenv("LORA_SCALE", "1.0"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.0"))
    id_loss_weight: float = float(os.getenv("ID_LOSS_WEIGHT", "0.35"))
    face_det_min_size: int = int(os.getenv("FACE_DET_MIN_SIZE", "80"))
    fps: int = int(os.getenv("SVD_FPS", "6"))
    motion_bucket_id: int = int(os.getenv("SVD_MOTION_BUCKET_ID", "127"))
    noise_aug_strength: float = float(os.getenv("SVD_NOISE_AUG", "0.01"))
    pose_sigma: float = float(os.getenv("POSE_SIGMA", "6.0"))
    body_crops_prefix: str = "body_crops"
    face_crops_prefix: str = "face_crops"
    pose_prefix: str = "pose_keypoints"
    models_prefix: str = "models/mimic_motion"
    identity_name: str = "identity.png"
    svd_dirname: str = "stable-video-diffusion-img2vid-xt-1-1"
    mm_unet_class: str = os.getenv("MIMICMOTION_UNET_CLASS", "").strip()

# -------------------------
# GCS & Data Handling
# -------------------------
def gcs(): return storage.Client()

def gcs_download_dir(bucket: str, prefix: str, local_dir: Path):
    t0 = time.time(); local_dir.mkdir(parents=True, exist_ok=True); n = 0
    for blob in gcs().list_blobs(bucket, prefix=prefix):
        rel = Path(blob.name).relative_to(prefix)
        if str(rel) == ".": continue
        dst = local_dir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst)); n += 1
    jlog("gcs_download_dir", bucket=bucket, prefix=prefix, local=str(local_dir), files=n, secs=round(time.time()-t0, 2))

def gcs_upload_dir(local_dir: Path, bucket: str, prefix: str):
    t0 = time.time(); b = gcs().bucket(bucket); up = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            dest = f"{prefix}/{p.relative_to(local_dir).as_posix()}"
            b.blob(dest).upload_from_filename(str(p)); up += 1
    jlog("gcs_upload_dir", bucket=bucket, prefix=prefix, local=str(local_dir), files=up, secs=round(time.time()-t0, 2))

def load_rgb(path: Path, target_hw: Tuple[int,int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (target_hw[1], target_hw[0]): img = img.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
    return np.array(img)

def to_tensor_image(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(u8).float().div(255.0).permute(2,0,1).mul(2.0).sub(1.0)

def pose_json_to_heatmaps(pose_json: dict, H: int, W: int, sigma: float) -> np.ndarray:
    import numpy as _np; import cv2 as _cv2
    H, W = int(H), int(W); heat = _np.zeros((H, W), dtype=_np.float32)
    kps = pose_json.get("keypoints", None) or pose_json.get("pose_keypoints_2d", None)
    if not isinstance(kps, list) or len(kps) == 0: return heat[None, ...]
    pts = []
    try:
        if isinstance(kps[0], (int, float, _np.number)):
            for i in range(0, len(kps), 3): pts.append((float(kps[i]), float(kps[i+1]), float(kps[i+2])))
        elif isinstance(kps[0], (list, tuple)):
            for p in kps: pts.append((float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 1.0))
        elif isinstance(kps[0], dict):
            for d in kps: pts.append((float(d.get("x",0)), float(d.get("y",0)), float(d.get("score",1.0))))
    except Exception: return heat[None, ...]
    if not pts: return heat[None, ...]
    max_x, max_y = max(p[0] for p in pts), max(p[1] for p in pts)
    is_normalized = (max_x <= 2.0 and max_y <= 2.0)
    src_size = pose_json.get("size", None)
    for (x, y, conf) in pts:
        if conf <= 0.0: continue
        if is_normalized: px, py = x * W, y * H
        else:
            if isinstance(src_size, (list, tuple)) and len(src_size) == 2:
                sx, sy = W / max(1.0, float(src_size[0])), H / max(1.0, float(src_size[1]))
                px, py = x * sx, y * sy
            else: px, py = x, y
        ix, iy = int(_np.clip(round(px), 0, W - 1)), int(_np.clip(round(py), 0, H - 1))
        _cv2.circle(heat, (ix, iy), max(1, int(sigma)), 1.0, -1)
    if heat.max() > 0: heat /= heat.max()
    return heat[None, ...]

class BodyPoseDataset(Dataset):
    def __init__(self, body_dir: Path, pose_dir: Path, H: int, W: int, sigma: float):
        self.paths = sorted([p for p in body_dir.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}])
        self.pose_dir, self.H, self.W, self.sigma = pose_dir, H, W, sigma
        assert self.paths, f"No body crops in {body_dir}"
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        ip = self.paths[idx]; name = ip.stem
        img = load_rgb(ip, (self.H, self.W)); img_t = to_tensor_image(img)
        pose = np.zeros((1, self.H, self.W), np.float32)
        pj = self.pose_dir / f"{name}.json"
        if pj.exists():
            try: pose = pose_json_to_heatmaps(json.load(open(pj,"r")), self.H, self.W, self.sigma)
            except Exception as e: jlog("pose_json_error", path=str(pj), error=str(e))
        return {"image": img_t, "pose": torch.from_numpy(pose), "name": name}

# -------------------------
# LoRA and Model Handling
# -------------------------
class LoRAInjectedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float, scale: float):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rank, self.alpha, self.scale = rank, alpha, scale
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False) if linear.bias is not None else None
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)); nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank
    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        l = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base + self.scale * self.scaling * l

def inject_lora_attn(model: nn.Module, rank: int, alpha: float, dropout: float, scale: float) -> int:
    patterns = [re.compile(p) for p in [r"\.to_q$", r"\.to_k$", r"\.to_v$", r"\.to_out\.0$", r"\.in_proj$", r"\.out_proj$", r"\.proj$", r"\.proj_in$", r"\.proj_out$"]]
    scope = re.compile(r"(attn|Attention|transformer|temporal)", re.IGNORECASE)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not scope.search(name): continue
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}"
            if isinstance(child, nn.Linear) and any(p.search(full) for p in patterns):
                setattr(module, child_name, LoRAInjectedLinear(child, rank, alpha, dropout, scale)); replaced += 1
            if isinstance(child, nn.Sequential):
                for i, sub in enumerate(child):
                    if isinstance(sub, nn.Linear) and any(p.search(f"{full}.{i}") for p in patterns):
                        child[i] = LoRAInjectedLinear(sub, rank, alpha, dropout, scale); replaced += 1
    jlog("lora_injection_done", replaced=replaced); return replaced

def lora_params(model: nn.Module): return [p for n, p in model.named_parameters() if "lora_" in n]

def save_lora_safetensors(model: nn.Module, path: Path):
    from safetensors.torch import save_file
    state = {f"{n}.lora_A": m.lora_A.detach().cpu() for n, m in model.named_modules() if isinstance(m, LoRAInjectedLinear)}
    state.update({f"{n}.lora_B": m.lora_B.detach().cpu() for n, m in model.named_modules() if isinstance(m, LoRAInjectedLinear)})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path)); jlog("checkpoint_saved", path=str(path))

def instantiate_mimicmotion_unet(config_dir: Path, weights_file: Path, override_class: str = "") -> nn.Module:
    mm = importlib.import_module("mimicmotion.modules.unet")
    class_name = override_class or "UNetSpatioTemporalConditionModel"
    if not hasattr(mm, class_name): raise RuntimeError(f"Cannot find '{class_name}' in mimicmotion.modules.unet.")
    cls = getattr(mm, class_name)
    unet = cls.from_config(config_dir.as_posix())
    jlog("mimicmotion_unet_instantiated_from_config", config_dir=config_dir, in_channels=unet.config.in_channels)
    sd = torch.load(weights_file, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    jlog("mimicmotion_weights_loaded", missing=len(missing), unexpected=len(unexpected))
    return unet

# -------------------------
# AI / ML Helpers
# -------------------------
class IdentityEmbedder:
    def __init__(self, device: torch.device, det_min_size: int = 80):
        import insightface
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in set(ort.get_available_providers()): providers = ["CPUExecutionProvider"]
        except Exception: providers = ["CPUExecutionProvider"]
        self.device = device
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if device.type=="cuda" and "CUDAExecutionProvider" in providers else -1, det_size=(640,640))
    @torch.no_grad()
    def embed(self, img_rgb_u8: np.ndarray) -> Optional[np.ndarray]:
        faces = sorted(self.app.get(img_rgb_u8), key=lambda f: f.bbox[2]*f.bbox[3], reverse=True)
        return faces[0].normed_embedding.astype(np.float32) if faces else None

def set_seed(seed: int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def get_autocast_dtype(p: str): return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(p.lower())
def latent_hw_from_img(H: int, W: int) -> Tuple[int,int]: return H // 8, W // 8
def get_added_time_ids(B: int, fps: int, motion_id: int, noise_aug: float, device, dtype):
    return torch.tensor([fps, motion_id, noise_aug], device=device, dtype=dtype).expand(B, -1)
def grad_global_norm(parameters):
    params = [p for p in parameters if p.grad is not None]
    if not params: return 0.0
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in params]), 2.0).item()
def cuda_mem():
    if not torch.cuda.is_available(): return {}
    return {"mem_alloc_mb": round(torch.cuda.memory_allocated()/(1024**2),2), "mem_resv_mb": round(torch.cuda.memory_reserved()/(1024**2),2)}

# -------------------------
# Main Training Logic
# -------------------------
def main():
    cfg = TrainConfig(jobs_bucket=os.getenv("JOBS_BUCKET"), avatar_id=os.getenv("AI_AVATAR_ID"))
    assert cfg.jobs_bucket and cfg.avatar_id, "Set JOBS_BUCKET and AI_AVATAR_ID"
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_autocast_dtype(cfg.mixed_precision)
    jlog("env", device=str(device), mixed_precision=str(amp_dtype), torch_version=torch.__version__)

    # --- Setup Directories & Data ---
    root = cfg.local_root; data_root = root / "data"; models_root = root / "models"; out_root = root / "out"
    ckpt_dir = out_root / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    body_dir = data_root / "body_crops"; face_dir = data_root / "face_crops"; pose_dir = data_root / "pose_keypoints"
    gcs_download_dir(cfg.jobs_bucket, f"ai_avatars/{cfg.avatar_id}/{cfg.body_crops_prefix}", body_dir)
    gcs_download_dir(cfg.jobs_bucket, f"ai_avatars/{cfg.avatar_id}/{cfg.face_crops_prefix}", face_dir)
    gcs_download_dir(cfg.jobs_bucket, f"ai_avatars/{cfg.avatar_id}/{cfg.pose_prefix}", pose_dir)
    gcs_download_dir(cfg.jobs_bucket, cfg.models_prefix, models_root)

    # --- Load Models ---
    svd_dir = models_root / cfg.svd_dirname
    pipe = StableVideoDiffusionPipeline.from_pretrained(svd_dir.as_posix(), torch_dtype=torch.float16 if amp_dtype else torch.float32)
    vae = pipe.vae.to(device, dtype=torch.float16 if amp_dtype else torch.float32)
    image_encoder = pipe.image_encoder.to(device=device, dtype=torch.float32).eval()
    
    # Step 1: Instantiate the UNet on the CPU from the correct SVD unet config.
    unet = instantiate_mimicmotion_unet(
        config_dir=svd_dir / "unet",
        weights_file=models_root / "MimicMotion_1-1.pth",
        override_class=cfg.mm_unet_class
    )
    
    # Step 2: Inject the LoRA layers (also on CPU).
    assert inject_lora_attn(unet, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_scale) > 0
    
    # Step 3: Move the entire modified model to the GPU.
    unet = unet.to(device)
    
    # Step 4: Freeze base and activate LoRA grads.
    for p in unet.parameters(): p.requires_grad = False
    for p in lora_params(unet): p.requires_grad = True

    # --- Dataset and Identity ---
    ds = BodyPoseDataset(body_dir, pose_dir, cfg.crop_h, cfg.crop_w, cfg.pose_sigma)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    id_embedder = IdentityEmbedder(device, cfg.face_det_min_size)
    identity_path = face_dir / cfg.identity_name
    assert identity_path.exists(), f"Identity image not found at {identity_path}"
    id_emb_np = id_embedder.embed(load_rgb(identity_path, (cfg.crop_h, cfg.crop_w)))
    assert id_emb_np is not None, "Could not detect a face in the identity image."
    identity_emb = F.normalize(torch.from_numpy(id_emb_np).to(device), dim=0)

    # --- Core ML Functions ---
    @torch.no_grad()
    def encode_imgs(imgs: torch.Tensor) -> torch.Tensor:
        return vae.encode(imgs.to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(lat: torch.Tensor, num_frames: int) -> torch.Tensor:
        lat = lat.to(dtype=vae.dtype)
        return vae.decode(lat / vae.config.scaling_factor, num_frames=num_frames).sample

    @torch.no_grad()
    def svd_image_embeds(rgb_t: torch.Tensor) -> torch.Tensor:
        pil_list = [Image.fromarray(arr) for arr in (rgb_t.clamp(-1,1).add(1).mul(127.5).round().byte().permute(0,2,3,1).cpu().numpy())]
        pixel_values = pipe.feature_extractor(images=pil_list, return_tensors="pt")["pixel_values"]
        vision_output = image_encoder(pixel_values.to(device, torch.float32))
        image_embeds = vision_output.image_embeds
        return image_embeds.unsqueeze(1)

    def cosine_id_metrics(imgs_decoded: torch.Tensor) -> Dict[str, float]:
        np_imgs = ((imgs_decoded.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
        sims = [torch.dot(F.normalize(torch.from_numpy(emb).to(device), dim=0), identity_emb).item() for emb in [id_embedder.embed(im) for im in np_imgs] if emb is not None]
        return {"cos_sim_mean": np.mean(sims) if sims else 0.0, "faces": len(sims)}

    # --- Optimizer & Scheduler ---
    optim = torch.optim.AdamW(lora_params(unet), lr=cfg.lr, weight_decay=cfg.weight_decay)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    amp_enabled = amp_dtype is not None

    # --- Training Loop ---
    global_step, t_train0 = 0, time.time()
    for epoch in range(cfg.epochs):
        unet.train(); jlog("epoch_start", epoch=epoch+1, epochs=cfg.epochs, **cuda_mem())
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for step, batch in enumerate(pbar):
            step_start_time = time.time()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                imgs_bchw = batch["image"].to(device, non_blocking=True)
                pose1_bchw = batch["pose"].to(device, non_blocking=True).clamp(0,1)
                B = imgs_bchw.size(0)

                latents_bchw = encode_imgs(imgs_bchw)
                noise = torch.randn_like(latents_bchw)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
                noisy_latents_bchw = noise_scheduler.add_noise(latents_bchw, noise, timesteps)
                
                h_lat, w_lat = latents_bchw.shape[-2:]
                pose_interp_bchw = F.interpolate(pose1_bchw, size=(h_lat, w_lat), mode="bilinear", align_corners=False)
                pose_4ch_bchw = pose_interp_bchw.repeat(1, 4, 1, 1)

                sample_8ch_bchw = torch.cat([noisy_latents_bchw, pose_4ch_bchw], dim=1)
                sample_bfchw = sample_8ch_bchw.unsqueeze(1)
                pose_320ch_bchw = pose_4ch_bchw.repeat_interleave(80, dim=1)

                encoder_hidden_states = svd_image_embeds(imgs_bchw)
                added_time_ids = get_added_time_ids(B, cfg.fps, cfg.motion_bucket_id, cfg.noise_aug_strength, device, encoder_hidden_states.dtype)
                
                pred_bfchw = unet(
                    sample=sample_bfchw, timestep=timesteps, encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids, pose_latents=pose_320ch_bchw,
                    image_only_indicator=False, return_dict=False
                )[0]
                
                pred_noise_bchw = pred_bfchw.squeeze(1)[:, :4]
                loss_mse = F.mse_loss(pred_noise_bchw, noise)

            with torch.no_grad():
                pred_latents = noise_scheduler.step(pred_noise_bchw, timesteps[0], noisy_latents_bchw).pred_original_sample
                imgs_dec_1f = decode_latents(pred_latents, num_frames=1)
            
            id1 = cosine_id_metrics(imgs_dec_1f)
            id_factor = 1.0 + cfg.id_loss_weight * max(0.0, 1.0 - id1["cos_sim_mean"])
            
            loss_total = loss_mse * id_factor

            scaler.scale(loss_total / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                grad_norm = grad_global_norm(lora_params(unet))
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True); global_step += 1
                
                jlog(
                    "train_step", step=global_step, epoch=epoch+1,
                    loss=loss_total.item(), mse=loss_mse.item(), 
                    id1_sim=id1["cos_sim_mean"],
                    id_factor=id_factor, grad_norm=grad_norm, lr=optim.param_groups[0]['lr'],
                    time_per_step=time.time() - step_start_time, **cuda_mem()
                )

            if cfg.max_train_steps and global_step >= cfg.max_train_steps: break
        if cfg.max_train_steps and global_step >= cfg.max_train_steps: break

        save_lora_safetensors(unet, ckpt_dir / f"epoch_{epoch+1:02d}.safetensors")
        jlog("epoch_end", epoch=epoch+1, **cuda_mem())

    save_lora_safetensors(unet, ckpt_dir / "final.safetensors")
    out_prefix = f"ai_avatars/{cfg.avatar_id}/models/lora_mimicmotion_unet"
    gcs_upload_dir(ckpt_dir, cfg.jobs_bucket, out_prefix)
    jlog("training_done", total_secs=round(time.time()-t_train0, 2), final_prefix=f"gs://{cfg.jobs_bucket}/{out_prefix}")
    print("✅ Training complete.")

if __name__ == "__main__":
    main()