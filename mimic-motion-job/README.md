# Build image
```
gcloud builds submit --tag gcr.io/billion-ai-girls/mimic-motion-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=mimic-motion-job \
  --config=<(echo '{
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "a2-ultragpu-1g",
          "acceleratorType": "NVIDIA_A100_80GB",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "diskSpec": {
          "bootDiskType": "pd-ssd",
          "bootDiskSizeGb": 200
        },
        "containerSpec": {
          "imageUri": "gcr.io/billion-ai-girls/mimic-motion-job:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-jobs"
            },
            {
              "name": "EXECUTION_ID",
              "value": "9d891abe-3f41-412b-b72c-98d12995fe18"
            },
            {
              "name": "REF_IMAGE_PATH",
              "value": "jobs/9d891abe-3f41-412b-b72c-98d12995fe18/image2.png"
            },
            {
              "name": "REF_VIDEO_PATH",
              "value": "jobs/9d891abe-3f41-412b-b72c-98d12995fe18/video2.mp4"
            }
          ]
        }
      }
    ]
  }')
```

# Local test

```
docker build -t mimic-motion-job:dev . --no-cache

docker run --rm \
  -e EXECUTION_ID=9d891abe-3f41-412b-b72c-98d12995fe18 \
  -e JOBS_BUCKET=billion-ai-girls-jobs \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcreds.json \
  -v ./gcreds.json:/gcreds.json:ro \
  mimic-motion-job:dev
```