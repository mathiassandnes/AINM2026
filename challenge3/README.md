# Challenge 3: Astar Island — Norse World Prediction

## Overview
Observe a Norse world simulator through limited viewports and predict terrain probability distributions across a 40×40 map after 50 years of simulation.

## Format
- **Submission:** REST API calls to their platform
- **Timeout:** 60 seconds per round
- **Query budget:** 50 queries total per round, shared across 5 seeds

## Terrain Types (6 classes)
| Class | Terrain |
|-------|---------|
| 0 | Ocean, Plains, Empty |
| 1 | Settlements |
| 2 | Ports |
| 3 | Ruins |
| 4 | Forests |
| 5 | Mountains (static) |

## Simulation Phases (yearly cycle)
1. **Growth** — settlements expand?
2. **Conflict** — settlements fight/destroy?
3. **Trade** — ports matter?
4. **Winter** — decay/death?
5. **Environment reclamation** — forests regrow?

## Query System
- 50 queries total per round, shared across 5 seeds
- Each query reveals max **15×15 viewport** of full **40×40 map**
- Must strategically choose WHERE to look

### Query Math
- 50 queries / 5 seeds = **10 queries per seed**
- 15×15 = 225 cells per query, map = 1600 cells
- 10 queries × 225 = 2250 cells visible per seed (with overlap)
- You CAN cover the full map per seed with ~8 non-overlapping queries (8 × 225 = 1800 > 1600)
- That leaves 50 - 40 = 10 spare queries for targeted re-observation

## Prediction Format
- 3D array: `prediction[y][x][class]` — 40×40×6
- Each cell's 6 probabilities must sum to 1.0 (±0.01)
- **CRITICAL: Never assign 0.0 to any class!** KL divergence → ∞ on zeros.
- Use minimum **0.01 floor** per class, then renormalize.

## API Endpoints
```
GET  /astar-island/rounds              - List all rounds
GET  /astar-island/rounds/{round_id}   - Round details
GET  /astar-island/budget              - Remaining queries
POST /astar-island/simulate            - Run sim, reveal viewport
POST /astar-island/submit              - Submit predictions
GET  /astar-island/my-rounds           - Team-specific rounds
GET  /astar-island/my-predictions/{id} - Your predictions
GET  /astar-island/leaderboard         - Public leaderboard
```

## Scoring
- **Metric:** Entropy-weighted KL divergence
- **Score:** `100 × exp(-KL_divergence)`
- Only **dynamic cells** (high entropy) contribute meaningfully
- Per-round: average of 5 seed scores
- Leaderboard: **best round score ever**

## Strategy

### Approach 1: Full Coverage + Prior Learning
1. Use ~8 queries per seed to tile the full 40×40 map (non-overlapping 15×15 viewports)
2. That uses 40 of 50 queries — 10 left for targeted re-checks
3. Observe terrain at year 50 across all 5 seeds
4. Where all seeds agree → high confidence
5. Where seeds disagree → spread probability across observed outcomes

### Approach 2: Learn the Simulation Rules
1. The 5-phase cycle is likely deterministic given initial conditions
2. Query early-year states AND late-year states to understand dynamics
3. Build a transition model: P(terrain_t+1 | terrain_t, neighbors)
4. Predict unobserved cells from observed neighbors

### Key Insights
- Mountains are **static** (class 5) — easy to predict with certainty
- Ocean likely static (class 0) — easy
- The interesting predictions are settlements, ports, ruins, forests
- Ruins = former settlements after conflict?
- Ports = settlements near ocean?
- Forests reclaim empty land?
- **Scoring weights dynamic cells more** — focus on getting the changing tiles right

### Minimum Probability Floor
```python
def safe_prediction(probs, floor=0.01):
    probs = np.maximum(probs, floor)
    probs /= probs.sum()
    return probs
```

## Priority: HIGH (most fun + interesting ML problem)
Combines exploration strategy, probabilistic prediction, and simulation understanding. RL background from masters directly relevant.
