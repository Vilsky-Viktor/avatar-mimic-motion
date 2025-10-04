# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/mimic-motion-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/mimic-motion-job:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-asia"
            },
            {
              "name": "EXECUTION_ID",
              "value": "fbe5afcb-4666-4933-b472-269ae858a597"
            },
            {
              "name": "REF_IMAGE_PATH",
              "value": "jobs/fbe5afcb-4666-4933-b472-269ae858a597/image1-1.png"
            },
            {
              "name": "REF_VIDEO_PATH",
              "value": "jobs/fbe5afcb-4666-4933-b472-269ae858a597/video1.mp4"
            },
            {
              "name": "LORA_RANK",
              "value": "64"
            },
            {
              "name": "LORA_SCALE",
              "value": "1.0"
            },
            {
              "name": "GUIDANCE_SCALE",
              "value": "2.5"
            }
          ],
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