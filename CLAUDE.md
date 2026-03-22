# NM i AI 2026 - Competition Workspace

## Competition
- **Event**: NM i AI (Norwegian Championship in AI), March 19-22, 2026
- **Participant**: Mathi, competing solo
- **Site**: https://app.ainm.no/

## Project Structure
- `challenge1/`, `challenge2/`, `challenge3/` — each challenge gets its own directory
- `shared/` — reusable utilities across challenges
- `notebooks/` — exploratory notebooks

## Tech Stack
- Python 3.13, package manager: `uv`
- Key libraries: see pyproject.toml
- Claude API available via `anthropic` SDK

## Conventions
- Keep solutions pragmatic — competition code, not production code
- Each challenge dir should have its own README with problem statement and approach notes
- Use notebooks for exploration, scripts for final submissions
- Prefer fast iteration over perfect architecture
