# Deploy workflow

```
gcloud workflows deploy ai-avatar-training-pipeline \
  --source=workflow.yaml \
  --location=us-central1
```

# Run workflow (example)

```
gcloud workflows run ai-avatar-training-pipeline \
  --data='{"ai_avatar_id": "shlyshka"}'
```