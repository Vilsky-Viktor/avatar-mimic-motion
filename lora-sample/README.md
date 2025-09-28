# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/lora-sample
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=lora-sample \
  --config=<(echo '{
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "a2-ultragpu-1g",
          "acceleratorType": "NVIDIA_A100_80GB",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "asia.gcr.io/billion-ai-girls/lora-sample:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-asia"
            },
            {
              "name": "AI_AVATAR_ID",
              "value": "ver1"
            },
            {
              "name": "TARGET_MODEL_NAME",
              "value": "white-socks"
            }
          ]
        }
      }
    ]
  }')
```