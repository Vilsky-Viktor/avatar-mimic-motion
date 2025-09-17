import io, json, cv2, numpy as np
from google.cloud import storage

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