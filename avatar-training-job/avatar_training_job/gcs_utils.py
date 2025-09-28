import io
from pathlib import Path
from typing import Iterable, List
from google.cloud import storage

def get_client():
    return storage.Client()

def bucket(project_bucket: str):
    return get_client().bucket(project_bucket)

def list_blobs(bucket, prefix: str, suffixes: Iterable[str] = ()):
    blobs = []
    for b in bucket.list_blobs(prefix=prefix):
        if not suffixes:
            blobs.append(b.name)
        else:
            low = b.name.lower()
            if any(low.endswith(s) for s in suffixes):
                blobs.append(b.name)
    return sorted(blobs)

def download_to_path(bucket, blob_name: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(str(dst))

def upload_bytes(bucket, blob_name: str, data: bytes, content_type: str):
    bucket.blob(blob_name).upload_from_file(io.BytesIO(data), content_type=content_type)

def upload_file(bucket, blob_name: str, src: Path, content_type: str = None):
    bucket.blob(blob_name).upload_from_filename(str(src), content_type=content_type)

def ensure_local_dataset(bucket, src_prefix: str, dst_dir: Path, suffixes: Iterable[str]) -> List[Path]:
    """Mirrors matching files from GCS to local dir and returns local paths."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = list_blobs(bucket, src_prefix, suffixes)
    paths = []
    for n in names:
        local = dst_dir / Path(n).name
        if not local.exists():
            download_to_path(bucket, n, local)
        paths.append(local)
    return paths

def download_folder(bucket, gcs_prefix: str, local_dir: Path):
    norm_prefix = gcs_prefix.rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=norm_prefix):
        rel = Path(blob.name).relative_to(norm_prefix)
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        download_to_path(bucket, blob.name, dst)