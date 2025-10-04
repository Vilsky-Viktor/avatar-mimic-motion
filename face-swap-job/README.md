# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/face-swap-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=face-swap-job \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/face-swap-job:latest",
          "env": [
            {
              "name": "VIDEO_URL",
              "value": "gs://billion-ai-girls-asia/jobs/fbe5afcb-4666-4933-b472-269ae858a597/generated.mp4"
            },
            {
              "name": "IMAGE_URL",
              "value": "gs://billion-ai-girls-asia/ai_avatars/ver1/body_crops/identity.png"
            },
          ]
        }
      }
    ]
  }')
```