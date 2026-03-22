#!/bin/bash
# Show task traces from Cloud Logging
# Usage: ./show_traces.sh [revision] [limit]
#   ./show_traces.sh              # latest revision, last 10
#   ./show_traces.sh 00030-l72 20 # specific revision, last 20

REV="${1:-}"
LIMIT="${2:-10}"
PROJECT="ainm26osl-784"

if [ -n "$REV" ]; then
    FILTER="resource.labels.service_name=\"tripletex-agent\" resource.labels.revision_name=\"tripletex-agent-${REV}\" jsonPayload.message=~\"TASK TRACE\""
else
    FILTER="resource.labels.service_name=\"tripletex-agent\" jsonPayload.message=~\"TASK TRACE\""
fi

gcloud logging read "$FILTER" \
    --limit "$LIMIT" \
    --format json \
    --project "$PROJECT" \
    --freshness=6h 2>/dev/null | python3 -c "
import json, sys
logs = json.load(sys.stdin)
for entry in reversed(logs):
    msg = entry.get('jsonPayload', {}).get('message', '')
    rev = entry.get('resource', {}).get('labels', {}).get('revision_name', '?')
    ts = entry.get('timestamp', '')[:19]
    print(f'\n[{ts}] [{rev}]')
    print(msg)
    print()
"
