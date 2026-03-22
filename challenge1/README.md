# Challenge 1: Tripletex — AI Accounting Agent

## Overview
Build an HTTPS endpoint that receives accounting task prompts, interprets instructions, calls the Tripletex API via proxy, and completes tasks (employee creation, invoicing, expense reporting, etc.).

## Format
- **Submission:** Deploy an HTTPS endpoint with `POST /solve`
- **Timeout:** 300 seconds (5 min) per task
- **Tasks:** 30 unique task types across 3 difficulty tiers
- **Languages:** Tasks span 7 languages

## Endpoint Contract
- Method: `POST /solve`
- Content-Type: `application/json`
- Response: `{"status": "completed"}` with HTTP 200

## Authentication
- Basic Auth: username `0`, password = session token
- Optional API key as Bearer token
- Each submission gets a **fresh sandbox account**

## Scoring
- Field-by-field correctness verification (0–1 scale)
- **Tier multipliers:** Tier 1 = ×1, Tier 2 = ×2, Tier 3 = ×3
- **Efficiency bonus:** Up to 2× multiplier for perfect scores with minimal API calls
- Best score per task tracked independently

## Tier Release Schedule
- Tier 1: Competition start (Thursday 18:00)
- Tier 2: Early Friday
- Tier 3: Early Saturday

## Strategy
This is essentially "Claude Code with good context." The differentiator is:
1. Quality of Tripletex API context fed to the LLM
2. Robust error handling and retry logic
3. Efficiency (fewer API calls = higher multiplier)
4. Multi-language support (7 languages)

### Deployment
Use Google Cloud Run (free via competition partnership):
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/solve")
async def solve(request: dict):
    # Parse task, call Tripletex API, complete accounting task
    return {"status": "completed"}
```

## Priority: MEDIUM
Everyone will throw Claude/GPT at this. Hard to differentiate. Focus on getting a solid baseline, then move on.
