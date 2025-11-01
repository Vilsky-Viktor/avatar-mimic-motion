# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/upscaling-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=upscaling-job \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/upscaling-job:latest",
          "env": [
            {
              "name": "JOBS_BUCKET",
              "value": "billion-ai-girls-asia"
            },
            {
              "name": "INPUT_URI",
              "value": "gs://billion-ai-girls-asia/jobs/fbe5afcb-4666-4933-b472-269ae858a597/video1_interpolated.mp4"
            },
            {
              "name": "ESRGAN_WEIGHTS_URI",
              "value": "gs://billion-ai-girls-asia/models/real-esrgan/RealESRGAN_x2plus.pth"
            },
          ]
        }
      }
    ]
  }')
```