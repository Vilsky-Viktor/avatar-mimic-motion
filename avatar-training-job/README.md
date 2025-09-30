# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/avatar-training-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=avatar-training-job \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/avatar-training-job:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-asia"
            },
            {
              "name": "EXECUTION_ID",
              "value": "9d891abe-3f41-412b-b72c-98d12995fe18"
            },
            {
              "name": "AI_AVATAR_ID",
              "value": "ver1"
            },
            {
              "name": "TARGET_MODEL_NAME",
              "value": "white-socks"
            },
            {
              "name": "DEBUG_SVD",
              "value": "0"
            },
            {
              "name": "ID_LATENT_DOWNSCALE",
              "value": "2"
            },
                        {
              "name": "ACC_STEPS",
              "value": "8"
            },
                        {
              "name": "LR",
              "value": "3e-5"
            },
                        {
              "name": "WD",
              "value": "0.01"
            },
                        {
              "name": "LORA_ALPHA",
              "value": "32.0"
            }
          ]
        }
      }
    ]
  }')
```