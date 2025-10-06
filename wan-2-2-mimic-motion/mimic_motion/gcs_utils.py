from pathlib import Path
from google.cloud import storage
import mimetypes  # NEW

def _client() -> storage.Client:
    return storage.Client()

def download_folder(bucket_name: str, prefix: str, dest_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # normalize prefix
    if not prefix.endswith("/"):
        prefix += "/"

    blobs = bucket.list_blobs(prefix=prefix)

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        # compute relative path inside local dir
        relative_path = Path(blob.name).relative_to(prefix)
        local_path = dest_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {blob.name} -> {local_path}")
        blob.download_to_filename(local_path)

def download_blob(bucket: str, src: str, dst: Path):
    client = _client()
    b = client.bucket(bucket)
    blob = b.blob(src)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{src} not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))

def upload_blob(bucket: str, src: Path, dst: str, content_type: str | None = None):
    client = _client()
    b = client.bucket(bucket)
    blob = b.blob(dst)
    if content_type:
        blob.content_type = content_type
    blob.upload_from_filename(str(src))

# NEW: recursively upload a whole folder, preserving relative paths
def upload_folder(bucket: str, src_dir: str | Path, dst_prefix: str) -> int:
    """
    Upload all files in src_dir to gs://{bucket}/{dst_prefix}/<relative path>,
    returning the number of files uploaded.
    """
    client = _client()
    b = client.bucket(bucket)

    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist")

    if dst_prefix and not dst_prefix.endswith("/"):
        dst_prefix += "/"

    count = 0
    for p in src_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src_dir).as_posix()
        key = f"{dst_prefix}{rel}"
        blob = b.blob(key)

        ctype, _ = mimetypes.guess_type(p.name)
        if ctype:
            blob.content_type = ctype

        print(f"Uploading {p} -> gs://{bucket}/{key}")
        blob.upload_from_filename(str(p))
        count += 1

    return count