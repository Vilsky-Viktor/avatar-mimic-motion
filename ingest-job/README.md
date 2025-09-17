Injest job

# Build image
```
gcloud builds submit --tag gcr.io/billion-ai-girls/ai-avatar-training-ingest-job
```

# Create Cloud Run Job
```
gcloud run jobs create ai-avatar-training-ingest-job \
  --image gcr.io/billion-ai-girls/ai-avatar-training-ingest-job \
  --region us-central1
```

# Run Job Independently
```
gcloud run jobs execute ai-avatar-training-ingest-job \
  --region=us-central1 \
  --update-env-vars=PAYLOAD='{"ai_avatar_id": "shlyshka"}',JOBS_BUCKET=billion-ai-girls-jobs,EXECUTION_ID=9d891abe-3f41-412b-b72c-98d12995fe18 \
  --project=billion-ai-girls
```