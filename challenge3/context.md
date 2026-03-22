# Astar Island — Challenge Context

## The Problem

We observe a **black-box Norse civilisation simulator** through a limited viewport and must predict the final world state as a probability distribution.

The simulator runs a procedurally generated Norse world for **50 years** — settlements grow, factions clash, trade routes form, alliances shift.

## World Grid

- **Size:** 40x40 cells
- **8 grid values map to 6 prediction classes:**

| Grid Value | Terrain | Prediction Class |
|-----------|---------|-----------------|
| 0 | Empty | 0 |
| 10 | Ocean | 0 |
| 11 | Plains | 0 |
| 1 | Settlement | 1 |
| 2 | Port | 2 |
| 3 | Ruin | 3 |
| 4 | Forest | 4 |
| 5 | Mountain | 5 |

- Mountains are static (never change)
- Forests are mostly static but can reclaim ruined land

## Simulation Phases (per year, 50 years total)

1. **Growth:** Settlements produce food based on adjacent terrain. Develop ports along coastlines. Build longships.
2. **Conflict:** Settlements raid each other. Longships extend raiding range significantly.
3. **Trade:** Ports within range of each other can trade if not at war.
4. **Winter:** All settlements lose food. Can collapse from starvation.
5. **Environment:** Natural world slowly reclaims abandoned land.

## Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (owner_id).

- **Initial state** exposes only: position and port status
- **Simulate response** exposes: population, food, wealth, defense, has_port, alive, owner_id (this is by design per the API docs)

The **hidden parameters** are the simulation RULES — growth rates, conflict aggression, trade ranges, winter severity. These govern how the simulation plays out and vary per round.

## Round Structure

1. Admin creates a round with a **fixed map**, many **hidden parameters**, and **5 random seeds**
2. Same map, different stochastic outcomes per seed
3. We get the full initial state for all 5 seeds for free (grid + settlement positions only)
4. We have **50 queries total** (shared across all 5 seeds) to observe year-50 state through viewports
5. We submit a 40x40x6 probability distribution per seed
6. ~2h45m prediction window per round

## Ground Truth

**The organizers pre-compute ground truth by running Monte Carlo simulations with the true hidden parameters.** This produces a probability distribution for each cell.

Same initial state + same seed → run hundreds of times → empirical distribution of outcomes at year 50.

## Scoring

- **Metric:** Entropy-weighted KL divergence: KL(ground_truth || prediction)
- **Score:** 100 * exp(-weighted_mean_KL), range 0-100
- **Entropy weighting:** Static cells (near-zero entropy) contribute nothing. High-entropy cells (conflict zones, boundaries) dominate the score.
- **Forward KL asymmetry:** Underestimating a probability is ~150x worse than overestimating. Always err toward more spread predictions.
- **Per-round score:** Average of 5 seed scores
- **Leaderboard:** Best (round_score × round_weight) ever. Round weight = 1.05^round_number.
- **CRITICAL:** Never assign 0.0 probability → KL divergence goes to infinity.

## Simulate Endpoint (POST /astar-island/simulate)

Each call runs **one stochastic simulation** and reveals a viewport window of the result.

**Confirmed stochastic:** Each call uses a different random sim_seed. Same viewport queried twice gives DIFFERENT results.

**Request:**
- round_id, seed_index (0-4), viewport_x, viewport_y, viewport_w (5-15), viewport_h (5-15)

**Response:**
- `grid`: 2D terrain values in viewport only (uses grid value encoding above)
- `settlements`: Settlements within viewport with full stats (position, population, food, wealth, defense, has_port, alive, owner_id)
- `viewport`: actual bounds {x, y, w, h} (clamped to map edges)
- `width`, `height`: full map dimensions (40x40)
- `queries_used`, `queries_max`: budget status

**Rate limit:** 5 req/sec per team. Budget: 50 queries per round total.

## Query Math

- 50 queries total, 5 seeds → ~10 per seed
- 9 non-overlapping 15x15 tiles cover the full 40x40 map
- Each query gives ONE sample from the distribution — not the distribution itself
- Querying same spot 10 times gives 10 samples for those cells but nothing about the rest of the map

## Other API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /astar-island/rounds | List all rounds |
| GET | /astar-island/rounds/{id} | Round details + initial states (free) |
| GET | /astar-island/budget | Remaining query budget |
| POST | /astar-island/submit | Submit 40x40x6 prediction for one seed (can resubmit, last counts) |
| GET | /astar-island/my-rounds | Our submissions + scores + initial_grid (first seed only) |
| GET | /astar-island/my-predictions/{id} | Our predictions with argmax + confidence grids |
| GET | /astar-island/analysis/{round_id}/{seed_index} | Ground truth comparison (completed rounds only) |
| GET | /astar-island/leaderboard | Public leaderboard |

## Available Training Data

- 5 completed rounds × 5 seeds = 25 pairs of (initial_state, ground_truth_distribution)
- Ground truth is 40x40x6 probability distributions (from Monte Carlo simulations)
- Saved locally in `challenge3/data/training_data.npz`

## What We Know From Data Analysis

From analyzing completed rounds:
- Ground truth distributions show high variance — many cells have probability spread across 2-3 classes
- Mean ground truth entropy is ~0.5, with ~1200/1600 cells being dynamic (entropy > 0.1)
- **Except Round 3:** only 223-338 dynamic cells — hidden parameters vary dramatically per round
- Settlement positions in initial state are the strongest predictor of year-50 outcomes
- Proximity to settlements determines how likely empty/forest cells become settlements
- Ports only appear adjacent to ocean (grid value 10)
- Mountains are perfectly deterministic (always class 5, 100%)

## Evaluation Results (leave-one-round-out CV)

| Method | Avg Score |
|--------|-----------|
| Global marginal (same dist everywhere) | 65.9 |
| Per-class marginal (avg by initial terrain) | 86.3 |
| Lookup table (class + dist_to_settlement + adj_ocean) | 88.8 |

Round-to-round variance is huge: lookup scores 97 on Round 4 but 75 on Round 3. Hidden parameters matter a lot.

## Our Scores

- **Round 5:** Submitted single observations with 95% confidence → **2.89** (terrible — wrong approach entirely)
- **Round 6:** Submitted lookup table predictions → score pending

## Current Leaderboard (top 5, weighted scores)

1. Meme Dream Team: 113.92
2. Sekyr: 113.88
3. Algebros: 113.08
4. Simple Levling: 112.19
5. Matriks: 111.58
