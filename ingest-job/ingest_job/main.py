import json, os, uuid, sys
from datetime import datetime, timezone
from google.cloud import storage

def main():
    # Read payload from env var (passed by Workflow)
    raw_payload = os.environ.get("PAYLOAD", "{}")
    try:
        payload = json.loads(raw_payload)
    except Exception as e:
        print(f"[ERROR] Could not parse PAYLOAD: {e}")
        sys.exit(1)

    bucket_name = os.environ["JOBS_BUCKET"]
    execution_id = os.environ["EXECUTION_ID"]

    now = datetime.now(timezone.utc).isoformat()

    manifest = {
        "execution_id": execution_id,
        "created_at": now,
        "ai_avatar_id": payload.get("ai_avatar_id"),
    }

    manifest_path = f"jobs/{execution_id}/manifest.json"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(manifest_path).upload_from_string(
        json.dumps(manifest, indent=2),
        content_type="application/json"
    )

    print(json.dumps(manifest))

if __name__ == "__main__":
    main()