import torch
import os
import numpy as np
import trimesh
import imageio
from pathlib import Path
from tqdm import tqdm
from spectre.src.spectre import SPECTRE  # main class
from spectre.config import cfg as spectre_cfg

SPECTRE_MODELS_DIR = Path(os.environ["SPECTRE_MODELS_DIR"])

def download_spectre_models(bucket):
    """
    Download SPECTRE pretrained models from GCS bucket to local_dir.
    Expected: models/spectre/SPECTRE.pth
    """
    SPECTRE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # List all files under models/spectre/
    blobs = bucket.list_blobs(prefix="models/spectre/")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        local_path = SPECTRE_MODELS_DIR / Path(blob.name).name
        if not local_path.exists():
            print(f"[SPECTRE] Downloading {blob.name} -> {local_path}")
            blob.download_to_filename(local_path)
        else:
            print(f"[SPECTRE] Already exists: {local_path}")

    print(f"[SPECTRE] Models ready in {SPECTRE_MODELS_DIR}")


def run_spectre(images_dir: Path, out_dir: Path):
    """
    Run SPECTRE on all images in `images_dir` and save outputs to `out_dir`.

    Produces:
      {img_id}_mesh.obj
      {img_id}_expr.npz
      {img_id}_debug.png
      face_identity_avg.obj
      face_identity.pt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SPECTRE] Using device: {device}")

    # ---- Config ----
    spectre_cfg.pretrained_modelpath = str(Path(SPECTRE_MODELS_DIR) / "spectre_model.tar")
    spectre_cfg.model.flame_model_path = str(Path(SPECTRE_MODELS_DIR) / "generic_model.pkl")
    spectre_cfg.model.flame_lmk_embedding_path = str(Path(SPECTRE_MODELS_DIR) / "landmark_embedding.npy")
    spectre_cfg.model.tex_path = str(Path(SPECTRE_MODELS_DIR) / "FLAME_albedo_from_BFM.npz")
    spectre_cfg.device = device

    model = SPECTRE(spectre_cfg, device)

    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([*images_dir.glob("*.png"),
                          *images_dir.glob("*.jpg"),
                          *images_dir.glob("*.jpeg")])
    if not image_files:
        print(f"[SPECTRE] No images found in {images_dir}")
        return

    # ---- Load images ----
    images_list = []
    for img_path in image_files:
        img = imageio.imread(img_path).astype(np.float32)
        if img.ndim == 2:  # grayscale → RGB
            img = np.stack([img] * 3, axis=-1)
        img = img / 255.0
        images_list.append(img)

    # Convert to torch tensor in **NCHW** format
    images_array = torch.from_numpy(np.array(images_list)).float()  # (N,H,W,3)
    images_array = images_array.permute(0, 3, 1, 2).to(device)      # (N,3,H,W)

    # ---- Encode ----
    print(f"[SPECTRE] Encoding {len(image_files)} images...")
    codedict, initial_deca_exp, initial_deca_jaw = model.encode(images_array)

    codedict['exp'] = codedict['exp'] + initial_deca_exp
    codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + initial_deca_jaw

    # ---- Decode ----
    opdict, visdict = model.decode(codedict, vis_lmk=False, return_vis=True)

    all_shapes = []
    faces_ref = model.flame.faces_tensor.cpu().numpy()

    for idx, img_path in enumerate(tqdm(image_files, desc="Saving results")):
        img_name = img_path.stem

        # Extract codes
        single_code = {
            k: (v[idx].detach().cpu().numpy() if isinstance(v, torch.Tensor) else v[idx])
            for k, v in codedict.items()
            if v is not None and len(v) > idx
        }
        if initial_deca_exp is not None and idx < len(initial_deca_exp):
            single_code["initial_deca_exp"] = initial_deca_exp[idx].detach().cpu().numpy()
        if initial_deca_jaw is not None and idx < len(initial_deca_jaw):
            single_code["initial_deca_jaw"] = initial_deca_jaw[idx].detach().cpu().numpy()

        np.savez(out_dir / f"{img_name}_expr.npz", **single_code)

        # Mesh
        verts = opdict["verts"][idx]
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()

        trimesh.Trimesh(vertices=verts, faces=faces_ref, process=False)\
            .export(out_dir / f"{img_name}_mesh.obj")

        # Debug render (optional)
        try:
            rend = opdict["rendered_images"][idx]
            rend_img = (rend.detach().cpu().numpy() * 255).astype(np.uint8)
            if rend_img.ndim == 3 and rend_img.shape[0] in (1, 3):
                rend_img = np.transpose(rend_img, (1, 2, 0))
            imageio.imwrite(out_dir / f"{img_name}_debug_texture.png", rend_img)
        except Exception as e:
            print(f"[warn] Debug render failed for {img_name}: {e}")

        try:
            shape_rend = visdict["shape_images"][idx]
            shape_rend_img = (shape_rend.detach().cpu().numpy() * 255).astype(np.uint8)
            if shape_rend_img.ndim == 3 and shape_rend_img.shape[0] in (1, 3):
                shape_rend_img = np.transpose(shape_rend_img, (1, 2, 0))
            imageio.imwrite(out_dir / f"{img_name}_debug_shape.png", shape_rend_img)
        except Exception as e:
            print(f"[warn] Debug shape image failed for {img_name}: {e}")

        if "shape" in single_code:
            all_shapes.append(single_code["shape"].squeeze())

    # ---- Identity average ----
    if all_shapes:
        mean_shape = np.mean(np.stack(all_shapes), axis=0)
        torch.save(torch.from_numpy(mean_shape).float(), out_dir / "face_identity.pt")

        with torch.no_grad():
            verts, _, _ = model.flame(
                shape_params=torch.from_numpy(mean_shape).unsqueeze(0).to(device).float(),
                expression_params=torch.zeros(
                    (1, codedict["exp"].shape[1] if "exp" in codedict else 50), device=device
                ),
                pose_params=torch.zeros(
                    (1, codedict["pose"].shape[1] if "pose" in codedict else 6), device=device
                ),
            )
        verts = verts[0].cpu().numpy()
        trimesh.Trimesh(vertices=verts, faces=faces_ref, process=False)\
            .export(out_dir / "face_identity_avg.obj")

    print(f"[SPECTRE] Finished. Results in {out_dir}")