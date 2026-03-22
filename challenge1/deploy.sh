#!/bin/bash
# Deploy to Google Cloud Run
set -e

# Load .env file if ANTHROPIC_API_KEY is not already set
if [ -z "$ANTHROPIC_API_KEY" ]; then
  for envfile in ../.env .env; do
    if [ -f "$envfile" ]; then
      export $(grep -v '^#' "$envfile" | grep ANTHROPIC_API_KEY | xargs)
    fi
  done
fi

# Fail fast if key is missing
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Create a .env file or export it."
  exit 1
fi

PROJECT_ID="${GCP_PROJECT:-nmiai-2026}"
SERVICE_NAME="tripletex-agent"
REGION="europe-north1"

echo "Deploying $SERVICE_NAME to Cloud Run ($REGION)..."

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" \
  --set-env-vars "CLAUDE_MODEL=${CLAUDE_MODEL:-claude-haiku-4-5-20251001}"

echo "Done. Get URL with:"
echo "  gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'"
