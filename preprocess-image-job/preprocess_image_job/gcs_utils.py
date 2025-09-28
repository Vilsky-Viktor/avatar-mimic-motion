import io, json, cv2, numpy as np
from google.cloud import storage
from pathlib import Path

def get_client():
    return storage.Client()

def read_json(bucket, path):
    blob = bucket.blob(path)
    
    return json.loads(blob.download_as_bytes().decode("utf-8"))

def list_images(bucket, prefix):
    return sorted([
        b.name for b in bucket.list_blobs(prefix=prefix)
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def download_image(bucket, path):
    data = bucket.blob(path).download_as_bytes()
    arr = np.frombuffer(data, np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def upload_png(bucket, path, img):
    ok, buf = cv2.imencode(".png", img)

    if not ok:
        raise RuntimeError("Encode failed")
    
    bucket.blob(path).upload_from_file(io.BytesIO(buf.tobytes()), content_type="image/png")

def download_folder(bucket, blob_prefix: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)

    for blob in bucket.list_blobs(prefix=blob_prefix):
        if blob.name.endswith("/"):
            continue

        rel = Path(blob.name).relative_to(blob_prefix)
        dst = local_dir / rel

        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"[GCS] Downloading {blob.name} -> {dst}")
            blob.download_to_filename(str(dst))
        else:
            print(f"[GCS] Skipping (exists): {dst}")