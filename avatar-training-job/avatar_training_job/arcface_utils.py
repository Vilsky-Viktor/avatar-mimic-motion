from typing import Optional
import numpy as np
import cv2
import torch
import insightface

class ArcFaceID:
    def __init__(self, model_name: str = "buffalo_l", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        try:
            self.model = insightface.app.FaceAnalysis(name=model_name, providers=providers)
            self.model.prepare(ctx_id=0 if self.device == "cuda" else -1)
            self.is_ready = True
        except Exception as e:
            print(f"WARNING: ArcFace initialization failed: {e}. Identity loss will be skipped.")
            self.is_ready = False
            self.model = None

    def embed_bgr(self, bgr: np.ndarray) -> Optional[torch.Tensor]:
        if not self.is_ready or bgr is None or bgr.size == 0:
            return None
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[-1] == 4:
            bgr = bgr[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self.model.get(rgb)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        return torch.tensor(emb, dtype=torch.float32)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return (a * b).sum()