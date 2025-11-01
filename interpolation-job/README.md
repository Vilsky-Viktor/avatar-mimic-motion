# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/interpolation-job
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=interpolation-job \
  --config=<(echo '{
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "g2-standard-8",
          "acceleratorType": "NVIDIA_L4",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "asia.gcr.io/billion-ai-girls/interpolation-job:latest",
          "env": [
            {
              "name": "VIDEO_URL",
              "value": "gs://billion-ai-girls-asia/jobs/fbe5afcb-4666-4933-b472-269ae858a597/video1.mp4"
            },
            {
              "name": "EXP",
              "value": "1"
            },
            {
              "name": "SCALE",
              "value": "2.0"
            },
          ],
        }
      }
    ]
  }')
```