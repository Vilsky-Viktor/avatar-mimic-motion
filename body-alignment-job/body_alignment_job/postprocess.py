from pathlib import Path
import pickle, shutil, numpy as np
from typing import Optional

def find_demo_result_dir(repo_root: Path, video_file: str) -> Path:
    # SMPLest-X README: "Inference output will be saved in SMPLest-X/demo"
    # We collect *.pkl (SMPL-X params) and any rendered overlays there.
    return repo_root / "demo"

def map_outputs_to_crops(result_dir: Path, crops: list[Path]):
    pkls = sorted(result_dir.rglob("*.pkl"))
    # if too many files exist, align by count (sorted order)
    if len(pkls) >= len(crops):
        pkls = pkls[:len(crops)]
    return pkls

def write_per_image_outputs(
    out_dir: Path,
    crops: list[Path],
    pkls: list[Path],
    overlays_dir: Optional[Path]
):
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_imgs = sorted(list(overlays_dir.rglob("*.png"))) if overlays_dir else []
    for i, crop in enumerate(crops):
        base = crop.stem
        # PKL
        if i < len(pkls):
            shutil.copy2(pkls[i], out_dir / f"{base}_smplx.pkl")
        # DEMO overlay (best effort)
        dst_demo = out_dir / f"{base}_demo.png"
        if i < len(demo_imgs):
            shutil.copy2(demo_imgs[i], dst_demo)
        else:
            shutil.copy2(crop, dst_demo)

def mean_betas_from_pkls(pkls: list[Path]) -> np.ndarray:
    betas = []
    for p in pkls:
        with open(p, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        for key in ("betas", "shape", "shape_param", "beta"):
            if key in data:
                v = np.array(data[key]).reshape(-1)
                betas.append(v)
                break
    if not betas:
        raise RuntimeError("No betas/shape params found in SMPLest-X outputs")
    d = max(10, max(len(b) for b in betas))
    B = np.zeros((len(betas), d), dtype=np.float32)
    for i, b in enumerate(betas):
        n = min(d, len(b)); B[i, :n] = b[:n]
    return B.mean(axis=0)