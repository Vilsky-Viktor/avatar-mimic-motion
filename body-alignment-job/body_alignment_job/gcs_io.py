from pathlib import Path
from typing import Iterable
from google.cloud import storage

def client() -> storage.Client:
    return storage.Client()

def download_prefix(bucket: str, prefix: str, out_dir: Path, exts=(".png",)):
    c = client(); b = c.bucket(bucket)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for blob in b.list_blobs(prefix=prefix):
        if blob.name.endswith("/"): continue
        if exts and not blob.name.lower().endswith(exts): continue
        dst = out_dir / Path(blob.name).name
        blob.download_to_filename(str(dst))
        paths.append(dst)
    return sorted(paths)

def download_file(bucket: str, key: str, dst: Path):
    c = client(); b = c.bucket(bucket); bl = b.blob(key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    bl.download_to_filename(str(dst))

def upload_tree(bucket: str, root: Path, dst_prefix: str):
    c = client(); b = c.bucket(bucket)
    for p in root.rglob("*"):
        if p.is_file():
            b.blob(f"{dst_prefix}/{p.relative_to(root).as_posix()}") \
             .upload_from_filename(str(p))