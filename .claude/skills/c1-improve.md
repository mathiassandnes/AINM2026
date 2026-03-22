# Challenge 1: Observe, Diagnose, Improve, Deploy

Iterative improvement loop for the Challenge 1 Tripletex agent.

## Steps

### 1. Fetch recent traces
Run `cd /Users/mathi/Cowork/NMiAI/challenge1 && bash show_traces.sh` to get the latest task traces from Cloud Logging. If the user specifies a revision or count, pass those as arguments.

### 2. Analyze failures
Review the traces and identify:
- Tasks that failed (plan failures, fallback failures, API errors)
- Patterns across failures (same task type failing, same API call erroring, wrong parameters)
- Inefficiencies (too many API calls, unnecessary retries)
- Phase 3 execution errors that triggered Phase 4 fallback unnecessarily

Present a concise diagnosis to the user:
- What went wrong (with specific trace IDs and task types)
- Root cause analysis
- Suggested code changes (which files, what to change, why)

### 3. Wait for user approval
Ask the user if they want to proceed with the suggested changes. Let them pick which improvements to implement.

### 4. Implement changes
Make the code changes in the relevant files under `challenge1/`. Key files:
- `agent.py` — orchestration, phase logic
- `planner.py` — Phase 1/2 prompts, operation menu
- `executor.py` — plan execution, variable resolution
- `tools.py` — composite tool implementations
- `tripletex.py` — HTTP client
- `server.py` — task classification, logging
- `api_schemas.py` — API schemas for Phase 2

### 5. Deploy
Run `cd /Users/mathi/Cowork/NMiAI/challenge1 && bash deploy.sh` to deploy the updated agent to Cloud Run. Show the user the deployment output.
