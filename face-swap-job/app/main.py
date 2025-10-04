import json, os, re, shlex, subprocess, sys, tempfile, pathlib, time
from dataclasses import dataclass
from typing import Tuple
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------- Utilities --------

def log(severity: str, msg: str, **kw):
    rec = {"severity": severity.upper(), "message": msg, **kw}
    print(json.dumps(rec), flush=True)

class Fatal(Exception): pass

_GCS_RE = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<path>.+)$")

def parse_gs_uri(uri: str) -> Tuple[str, str]:
    m = _GCS_RE.match(uri or "")
    if not m:
        raise Fatal(f"Invalid GCS URI: {uri!r}")
    return m.group("bucket"), m.group("path")

@dataclass
class Inputs:
    video_gs: str
    image_gs: str

    @staticmethod
    def from_env():
        v = os.getenv("VIDEO_URL")
        i = os.getenv("IMAGE_URL")
        if not v or not i:
            raise Fatal("Missing VIDEO_URL or IMAGE_URL env variables.")
        return Inputs(v, i)

# -------- GCS I/O --------

client = storage.Client()  # ADC is auto-configured in Vertex AI.  [oai_citation:5‡Google Cloud](https://cloud.google.com/vertex-ai/docs/training/code-requirements?utm_source=chatgpt.com)

@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def gcs_download(bucket: str, path: str, local: str):
    b = client.bucket(bucket)
    blob = b.blob(path)
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local)
    return local

@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def gcs_upload(local: str, bucket: str, path: str):
    b = client.bucket(bucket)
    blob = b.blob(path)
    blob.upload_from_filename(local)
    return f"gs://{bucket}/{path}"

# -------- FaceFusion runner --------

def run_facefusion(image_path: str, video_path: str, out_path: str):
    ff = "/opt/facefusion/run.py"

    # Optional: prefetch models to reduce first-run overhead
    try:
        subprocess.run(
            ["python3.11", ff, "force-download", "--log-level", "info"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        # non-fatal: proceed anyway, downloads will happen lazily
        log("WARNING", "force-download failed; continuing", output=e.stdout)

    cmd = [
        "python3.11", ff, "headless-run",
        "--source-paths", image_path,
        "--target-path", video_path,
        "--output-path", out_path,
        "--face-swapper-model", os.getenv("FACE_SWAPPER_MODEL", "inswapper_128_fp16"),
        "--execution-providers", os.getenv("EXECUTION_PROVIDERS", "cuda"),
        "--log-level", os.getenv("LOG_LEVEL", "info"),
    ]
    log("INFO", "Running FaceFusion", command=" ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log("INFO", "FaceFusion output", output=p.stdout[-5000:])  # tail for brevity
    if p.returncode != 0 or not pathlib.Path(out_path).exists():
        raise Fatal("FaceFusion failed; see logs above.")

def main():
    t0 = time.time()
    try:
        inp = Inputs.from_env()
        v_bucket, v_path = parse_gs_uri(inp.video_gs)
        i_bucket, i_path = parse_gs_uri(inp.image_gs)

        video_name = pathlib.Path(v_path).name
        video_dir  = pathlib.Path(v_path).parent.as_posix()
        out_name   = f"face_swap_{video_name}"
        out_gs     = f"gs://{v_bucket}/{video_dir}/{out_name}" if video_dir != "." else f"gs://{v_bucket}/{out_name}"

        with tempfile.TemporaryDirectory(prefix="face-swap-") as work:
            local_video = os.path.join(work, "input" + pathlib.Path(video_name).suffix)
            local_image = os.path.join(work, "face"  + pathlib.Path(i_path).suffix)
            local_out   = os.path.join(work, out_name)

            log("INFO", "Downloading inputs", video=inp.video_gs, image=inp.image_gs)
            gcs_download(v_bucket, v_path, local_video)
            gcs_download(i_bucket, i_path, local_image)

            run_facefusion(local_image, local_video, local_out)

            log("INFO", "Uploading result", dest=out_gs)
            out_uri = gcs_upload(local_out, v_bucket,
                                 f"{video_dir}/{out_name}" if video_dir != "." else out_name)

            log("INFO", "Done", result_uri=out_uri, elapsed_sec=round(time.time() - t0, 2))
            # Also print a plain line in case logs parser differs
            print(out_uri)
    except Fatal as e:
        log("ERROR", str(e))
        sys.exit(2)
    except Exception as e:
        log("ERROR", "Unhandled error", error=repr(e))
        sys.exit(99)

if __name__ == "__main__":
    main()