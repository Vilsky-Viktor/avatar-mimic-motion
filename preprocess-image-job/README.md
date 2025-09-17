# Build image
```
gcloud builds submit --tag gcr.io/billion-ai-girls/ai-avatar-training-preprocess-image-job
```

Do not use a2-ultragpu-1g with NVIDIA_A100_80GB (overkill)

# Run job
```
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=ai-avatar-training-preprocess-image-job \
  --config=<(echo '{
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "g2-standard-8",
          "acceleratorType": "NVIDIA_L4",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "diskSpec": {
          "bootDiskType": "pd-ssd",
          "bootDiskSizeGb": 200
        },
        "containerSpec": {
          "imageUri": "gcr.io/billion-ai-girls/ai-avatar-training-preprocess-image-job:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-jobs"
            },
            {
              "name": "EXECUTION_ID",
              "value": "9d891abe-3f41-412b-b72c-98d12995fe18"
            }
          ]
        }
      }
    ]
  }')
```

# LOcal test

```
docker build -t ai-avatar-training-preprocess-image-job:dev .

docker run --rm \
  -e EXECUTION_ID=9d891abe-3f41-412b-b72c-98d12995fe18 \
  -e JOBS_BUCKET=billion-ai-girls-jobs \
  -e GOOGLE_APPLICATION_CREDENTIALS=../gcreds.json \
  -v ../gcreds.json:/gcreds.json:ro \
  ai-avatar-training-preprocess-image-job:dev
```