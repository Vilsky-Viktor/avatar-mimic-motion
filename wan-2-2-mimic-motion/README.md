# Build image
```
gcloud builds submit --tag asia.gcr.io/billion-ai-girls/wan-2-2-mimic-motion
```

# Run job
```
gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --display-name=wan-2-2-mimic-motion \
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
          "imageUri": "asia.gcr.io/billion-ai-girls/wan-2-2-mimic-motion:latest",
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
              "value": "jobs/fbe5afcb-4666-4933-b472-269ae858a597/image-1-2.png"
            },
            {
              "name": "REF_VIDEO_PATH",
              "value": "jobs/fbe5afcb-4666-4933-b472-269ae858a597/video1.mp4"
            },
            {
              "name": "MODE",
              "value": "retarget"
            },
            {
              "name": "WIDTH",
              "value": "720"
            },
            {
              "name": "HEIGHT",
              "value": "1280"
            },
          ],
        }
      }
    ]
  }')
```