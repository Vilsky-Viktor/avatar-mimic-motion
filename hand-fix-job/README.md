# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/hand-fix-job --timeout=90m
```

```
"machineType": "g2-standard-8",
"acceleratorType": "NVIDIA_L4",
"acceleratorCount": 1
```

```
"machineType": "a2-ultragpu-1g",
"acceleratorType": "NVIDIA_A100_80GB",
"acceleratorCount": 1
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=hand-fix-job \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/hand-fix-job:latest",
          "env": [
            {
              "name": "DRIVING_VIDEO",
              "value": "gs://billion-ai-girls-asia/jobs/fbe5afcb-4666-4933-b472-269ae858a597/video1.mp4"
            },
            {
              "name": "GENERATED_VIDEO",
              "value": "gs://billion-ai-girls-asia/jobs/fbe5afcb-4666-4933-b472-269ae858a597/generated.mp4"
            },
            {
              "name": "ENABLE_FULL_DEBUG",
              "value": "true"
            },
          ]
        }
      }
    ]
  }')
```