from pathlib import Path
import numpy as np, torch, trimesh, smplx

def tpose_mesh_from_betas(human_model_root: Path, betas: np.ndarray, gender="neutral") -> trimesh.Trimesh:
    model = smplx.create(
        model_path=str(human_model_root / "smplx"),
        model_type="smplx",
        gender=gender.upper(),
        use_pca=False
    )
    betas_t = torch.tensor(betas, dtype=torch.float32)[None, :]
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32)
    out = model(
        betas=betas_t,
        body_pose=zeros(1, model.NUM_BODY_JOINTS * 3),
        global_orient=zeros(1, 3),
        jaw_pose=zeros(1, 3),
        leye_pose=zeros(1, 3),
        reye_pose=zeros(1, 3),
        expression=zeros(1, model.num_expression_coeffs),
        return_verts=True
    )
    verts = out.vertices[0].cpu().numpy()
    faces = model.faces
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def save_mesh(mesh: trimesh.Trimesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))