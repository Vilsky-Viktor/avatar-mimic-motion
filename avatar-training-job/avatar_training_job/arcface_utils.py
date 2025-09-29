# arcface_utils.py
from typing import Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional deps
# -----------------------------------------------------------------------------
try:
    # Differentiable face embedder (512-D), pretrained on VGGFace2
    from facenet_pytorch import InceptionResnetV1
    HAS_FACENET = True
except Exception:
    HAS_FACENET = False

try:
    # ONNX-based detector (for anchors/debug only; NOT used in the grad path)
    import insightface
    HAS_INSIGHT = True
except Exception:
    HAS_INSIGHT = False


def _fixed_image_standardization(x: torch.Tensor) -> torch.Tensor:
    """
    Facenet-style "prewhiten" normalization, per-image.
    Expects x in [0,1], shape [B,3,H,W].
    """
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
    return (x - mean) / std


def _safe_center_face_crop(rgb01: torch.Tensor, frac: float = 0.65, top_bias: float = 0.15) -> torch.Tensor:
    """
    Heuristic, differentiable crop around the central/top region (where faces usually are).
    rgb01: [B,3,H,W] float in [0,1]
    Returns a view (no copy) so gradients flow back into the source tensor.
    """
    assert rgb01.ndim == 4 and rgb01.shape[1] == 3, "rgb01 must be [B,3,H,W]"
    b, _, h, w = rgb01.shape
    side = int(min(h, w) * frac)
    cy = int(h * (0.5 - top_bias * 0.5))  # bias the crop a bit towards the top
    cx = w // 2
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(h, y0 + side)
    x1 = min(w, x0 + side)
    return rgb01[:, :, y0:y1, x0:x1]


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class ArcFaceID(nn.Module):
    """
    Identity embedding helper.

    - Training (differentiable) path: `embed_torch()` / `forward()` uses a
      PyTorch face embedder (InceptionResnetV1) so gradients flow to the input
      image. The embedder parameters are frozen.

    - Debug/anchor (non-differentiable) path: `embed_bgr()` optionally uses
      InsightFace ONNX detector to get a 512-D normalized embedding from a
      NumPy BGR image. This is convenience only and never used in the grad path.
    """

    def __init__(self, model_name: str = "buffalo_l", device: Optional[str] = None):
        super().__init__()

        # Resolve device robustly
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # -------------------- Torch embedder (required for training) --------------------
        self.embedder: Optional[nn.Module] = None
        if HAS_FACENET:
            self.embedder = InceptionResnetV1(pretrained="vggface2", classify=False).to(self.device).eval()
            for p in self.embedder.parameters():
                p.requires_grad = False
        else:
            print("ERROR: facenet-pytorch not installed. `pip install facenet-pytorch` to enable identity training.")

        # -------------------- Optional ONNX detector (debug/anchors) --------------------
        self.detector = None
        if HAS_INSIGHT:
            try:
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device.type == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self.detector = insightface.app.FaceAnalysis(name=model_name, providers=providers)
                self.detector.prepare(ctx_id=0 if self.device.type == "cuda" else -1)
            except Exception as e:
                print(f"WARNING: ArcFace detector init failed: {e}")

        self.is_ready = self.embedder is not None

    # -------------------- Differentiable path (used during training) --------------------
    def forward(self, rgb01: torch.Tensor, do_crop: bool = True) -> torch.Tensor:
        return self.embed_torch(rgb01, do_crop=do_crop)

    def embed_torch(self, rgb01: torch.Tensor, do_crop: bool = True) -> torch.Tensor:
        """
        Compute 512-D face embeddings from a batch of RGB images in [0,1].

        Args:
            rgb01:  [B,3,H,W] float tensor in [0,1]
            do_crop: if True, applies a center/top-biased crop before resizing

        Returns:
            [B,512] L2-normalized embeddings (fp32). Gradients flow to rgb01.
        """
        if not self.is_ready:
            raise RuntimeError("Torch face embedder not available (facenet-pytorch not installed).")

        assert rgb01.ndim == 4 and rgb01.shape[1] == 3 and rgb01.is_floating_point(), \
            "rgb01 must be float tensor with shape [B,3,H,W] in [0,1]"

        x = rgb01
        if do_crop:
            x = _safe_center_face_crop(x)  # differentiable slicing

        # Facenet expects ~160x160 crops
        x = torch.clone(x)  # be safe against views when resizing on some backends
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        x = _fixed_image_standardization(x)

        # Ensure we run the embedder in fp32 on its own device, with AMP disabled for stability
        embedder_device = next(self.embedder.parameters()).device  # type: ignore[arg-type]
        with torch.autocast(device_type=embedder_device.type, enabled=False):
            emb = self.embedder(x.to(device=embedder_device, dtype=torch.float32))  # [B,512]

        # Return normalized embeddings to make cosine similarity robust
        return F.normalize(emb, p=2, dim=1)

    # -------------------- Non-differentiable helper (anchors/debug) --------------------
    @torch.no_grad()
    def embed_bgr(self, bgr: np.ndarray) -> Optional[torch.Tensor]:
        """
        Convenience embedding from a single BGR image (NumPy).
        If an ONNX detector is available, uses it. Otherwise falls back to the
        torch path with the heuristic crop.

        Returns:
            [512] L2-normalized tensor on self.device, or None if no face found.
        """
        if bgr is None or bgr.size == 0:
            return None

        # Detector path (preferred for anchors/debug)
        if self.detector is not None:
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            if bgr.shape[-1] == 4:
                bgr = bgr[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = self.detector.get(rgb)
            if not faces:
                return None
            # Largest face
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            emb = faces[0].normed_embedding  # already L2-normalized (float32)
            emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
            return F.normalize(emb_t, p=2, dim=0)  # keep consistent even if upstream changes

        # Fallback: run through differentiable path with heuristic crop
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
            emb_b = self.embed_torch(t, do_crop=True)  # [1,512], normalized
            return emb_b[0].detach()
        except Exception:
            return None