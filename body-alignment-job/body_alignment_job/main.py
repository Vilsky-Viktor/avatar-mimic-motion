import os, json, subprocess, shlex, logging
from pathlib import Path
import numpy as np, torch
import cv2

from body_alignment_job.gcs_io import download_prefix, download_file, upload_tree
from body_alignment_job.postprocess import find_demo_result_dir, map_outputs_to_crops, write_per_image_outputs, mean_betas_from_pkls
from body_alignment_job.smpl_utils import tpose_mesh_from_betas, save_mesh

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("body-alignment-job")

# env
JOBS_BUCKET = os.environ["JOBS_BUCKET"]
EXECUTION_ID = os.environ["EXECUTION_ID"]
MODEL_DIR = os.environ.get("SMPlEST_MODEL_DIR", "smplest_x_h")
FPS = int(os.environ.get("FPS", "4"))

# paths
APP = Path("/app")
REPO = APP / "SMPLest-X"
PRETRAINED = REPO / "pretrained_models" / MODEL_DIR
HUMAN_MODELS = REPO / "human_models" / "human_model_files"
DATASET = APP / "dataset" / "body_crops"
WORK = APP / "work"; WORK.mkdir(parents=True, exist_ok=True)
OUT = APP / "output"

def load_manifest() -> dict:
    from google.cloud import storage
    blob = storage.Client().bucket(JOBS_BUCKET).blob(f"jobs/{EXECUTION_ID}/manifest.json")
    return json.loads(blob.download_as_text())

def ensure_models_from_bucket():
    # SMPLest-X pretrained
    log.info("downloading smplest_x models")
    download_file(JOBS_BUCKET, f"models/smplest_x/smplest_x_h.pth.tar", PRETRAINED / "smplest_x_h.pth.tar")
    download_file(JOBS_BUCKET, f"models/smplest_x/config_base.py", PRETRAINED / "config_base.py")

    log.info("downloading vitpose model")
    download_file(JOBS_BUCKET, f"models/vitpose/vitpose_huge.pth", PRETRAINED / "vitpose_huge.pth")

    log.info("downloading SMPLX models")
    # SMPL-X (minimum neutral npz + common aux files)
    (HUMAN_MODELS / "smplx").mkdir(parents=True, exist_ok=True)
    for fn in [
        "SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz",
        "SMPLX_NEUTRAL.pkl", "SMPLX_to_J14.pkl",
        "SMPL-X__FLAME_vertex_ids.npy", "MANO_SMPLX_vertex_ids.pkl"
    ]:
        try:
            download_file(JOBS_BUCKET, f"models/smplx/{fn}", HUMAN_MODELS / "smplx" / fn)
        except Exception:
            pass

    log.info("downloading SMPL models")
    (HUMAN_MODELS / "smpl").mkdir(parents=True, exist_ok=True)
    for fn in [
        "SMPL_FEMALE.pkl", "SMPL_MALE.pkl", "SMPL_NEUTRAL.pkl"
    ]:
        try:
            download_file(JOBS_BUCKET, f"models/smpl/{fn}", HUMAN_MODELS / "smpl" / fn)
        except Exception:
            pass

def make_video_from_crops(crops: list[Path], out_mp4: Path, fps: int):
    seq = WORK / "seq"
    seq.mkdir(parents=True, exist_ok=True)
    # clean
    for f in seq.glob("*"): f.unlink()

    # copy crops → numbered jpg
    for i, p in enumerate(crops, 1):
        img = cv2.imread(str(p))
        dst = seq / f"{i:06d}.jpg"
        cv2.imwrite(str(dst), img)

    # ensure output directory exists
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(seq / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_mp4)
    ]
    subprocess.check_call(cmd)

def run_smplest_inference(video_path: Path, fps: int):
    # README: put the file under SMPLest-X/demo and call scripts/inference.sh
    demo_dir = REPO / "demo"; demo_dir.mkdir(parents=True, exist_ok=True)
    target = demo_dir / video_path.name
    if target.exists(): target.unlink()
    os.symlink(video_path, target)
    cmd = f"bash -lc 'cd {REPO} && sh scripts/inference.sh {MODEL_DIR} {target.name} {fps}'"
    logging.info(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    log.info("downloading manifest ...")
    manifest = load_manifest()
    avatar_id = manifest["ai_avatar_id"]
    log.info(f"[BodyAlignment] avatar_id={avatar_id}")

    # 1) data
    
    log.info("downloading identity photos ...")
    crops = download_prefix(
        JOBS_BUCKET,
        f"ai_avatars/{avatar_id}/identity_images",
        Path("/app/dataset/identity_images"),   # local folder to save downloads
        exts=(".png", ".jpg", ".jpeg")
    )
    if not crops: raise RuntimeError("Avatar photos found")
    log.info(f"Downloaded {len(crops)} images")

    # 2) make mp4
    log.info("composing video from photos ...")
    mp4 = WORK / f"{avatar_id}_bodycrops.mp4"
    make_video_from_crops(crops, mp4, FPS)

    # 3) models
    log.info("downloading models ...")
    ensure_models_from_bucket()

    # 4) run SMPLest-X official inference
    log.info("running smplest inference ...")
    run_smplest_inference(mp4, FPS)

    # 5) collect outputs → map to images
    log.info("mapping outputs and images ...")
    result_dir = find_demo_result_dir(REPO, mp4.name)
    pkls = map_outputs_to_crops(result_dir, crops)

    log.info("writing results ...")
    OUT.mkdir(parents=True, exist_ok=True)
    write_per_image_outputs(OUT, crops, pkls, overlays_dir=result_dir)

    # 6) aggregate shape → T-pose mesh
    log.info("calculating canonical body ...")
    betas = mean_betas_from_pkls(pkls)
    torch.save(torch.tensor(betas, dtype=torch.float32), OUT / "body_shape.pt")
    mesh = tpose_mesh_from_betas(HUMAN_MODELS, betas, gender="neutral")
    save_mesh(mesh, OUT / "canonical_body.obj")

    # 7) upload
    log.info("uploading results to bucket ...")
    dst = f"ai_avatars/{avatar_id}/aligned/body"
    upload_tree(JOBS_BUCKET, OUT, dst)
    log.info(f"Uploaded to gs://{JOBS_BUCKET}/{dst}/")

if __name__ == "__main__":
    main()