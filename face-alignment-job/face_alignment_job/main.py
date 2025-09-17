import os, json
from pathlib import Path
from google.cloud import storage
from face_alignment_job.spectre_utils import run_spectre, download_spectre_models

JOBS_BUCKET   = os.environ["JOBS_BUCKET"]
EXECUTION_ID  = os.environ["EXECUTION_ID"]

INPUT_GCS_PREFIX  = "ai_avatars/{avatar_id}/dataset/face_crops"
OUTPUT_GCS_PREFIX = "ai_avatars/{avatar_id}/aligned/face"

def get_bucket():
    return storage.Client().bucket(JOBS_BUCKET)

def download_face_crops(bucket, avatar_id, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = INPUT_GCS_PREFIX.format(avatar_id=avatar_id)
    blobs = bucket.list_blobs(prefix=prefix)
    count = 0
    for b in blobs:
        if b.name.endswith(".png"):
            dst = local_dir / Path(b.name).name
            b.download_to_filename(str(dst))
            count += 1
    return count

def upload_outputs(bucket, avatar_id, local_out: Path):
    prefix = OUTPUT_GCS_PREFIX.format(avatar_id=avatar_id)
    for p in local_out.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_out)
            blob = bucket.blob(f"{prefix}/{rel.as_posix()}")
            blob.upload_from_filename(str(p))

def main():
    print("getting bucket")
    bucket = get_bucket()
    manifest_blob = bucket.blob(f"jobs/{EXECUTION_ID}/manifest.json")
    manifest_json = manifest_blob.download_as_text()
    print(f"manifest {manifest_json}")
    avatar_id = json.loads(manifest_json)["ai_avatar_id"]
    
    work = Path("/tmp/spectre_job")
    imgs_dir = work / "imgs"
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading face crops")
    n = download_face_crops(bucket, avatar_id, imgs_dir)
    if n == 0:
        print("No face crops found.")
        return
    
    print(f"downloaded {n} face crops")
    
    print("download models")
    download_spectre_models(bucket)

    print("run spectre")
    run_spectre(imgs_dir, out_dir)

    upload_outputs(bucket, avatar_id, out_dir)
    print("✅ SPECTRE job done.")

if __name__ == "__main__":
    main()