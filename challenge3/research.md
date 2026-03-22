# Astar Island — Research Notes

## Core Problem

Predict a 40x40x6 probability distribution (6 terrain classes per cell) for each of 5 seeds, given the initial state. Scored by entropy-weighted KL divergence: `score = 100 * exp(-mean_KL)`.

## Key Scoring Properties

### Forward KL asymmetry
KL(p||q) — underestimating a probability is ~150x worse than overestimating.
- Predicting q=0.001 when truth is p=0.15 → penalty 0.75 nats
- Predicting q=0.15 when truth is p=0.001 → penalty 0.005 nats
- **Always err toward more diffuse/spread predictions**

### Score sensitivity
| Weighted avg KL | Score |
|:---|:---|
| 0.00 | 100.0 |
| 0.05 | 86.1 |
| 0.10 | 74.1 |
| 0.20 | 54.9 |
| 0.50 | 22.3 |
| 1.00 | 5.0 |

Even 0.1 nats costs 26 points. Marginal improvements at low KL are enormously valuable.

### Entropy weighting
Only dynamic cells (high entropy in ground truth) contribute to the score. Deterministic cells (mountains, deep ocean) contribute zero regardless of prediction. High-entropy cells (conflict zones, boundaries) dominate the score.

## Available Training Data

- 5 completed rounds × 5 seeds = 25 pairs of (initial_state, ground_truth_distribution)
- Ground truth available via `GET /astar-island/analysis/{round_id}/{seed_index}`
- Each ground truth is a 40x40x6 probability distribution (from ~200 simulation runs)

## Approach: Learn Distribution from Training Data

Since we can't reliably build a local simulator (hidden parameters change per round), we should learn the mapping from initial state → probability distribution directly from the 25 training examples.

### Feature Engineering (per cell)

Candidate features for predicting a cell's year-50 distribution:
- **Initial class** (one-hot encoded, 6 classes)
- **Neighborhood class counts** at radius 1, 2, 3 (how many empty, forest, settlement, etc. neighbors)
- **Distance to nearest settlement** (manhattan)
- **Distance to nearest ocean** (manhattan)
- **Distance to nearest mountain** (manhattan)
- **Distance to nearest forest** (manhattan)
- **Number of settlements within radius R** (e.g. R=3, R=5)
- **Is adjacent to ocean** (boolean — ports only appear here)
- **Position features** (distance to map edge, center distance)
- **Local settlement density** (settlements in 5x5 window)
- **Neighborhood diversity** (entropy of terrain types in neighborhood)

### Model Options (ranked by robustness for 25 training grids)

1. **Per-initial-class marginal averages** (0 parameters) — unbeatable baseline
2. **Lookup table** with binned features — what we have now, can be improved
3. **Gradient-boosted trees** (LightGBM) with cross-entropy loss — ~100 trees, max depth 4
4. **Small MLP** (features → 64 → 32 → 6 softmax) — with dropout and weight decay
5. **Spatial CNN** (U-Net style) — only with aggressive augmentation (rotations, flips = 8x data)

With only 25 grid-level examples (40,000 cell-level examples), overfitting is the main risk. Use leave-one-grid-out cross-validation (25-fold).

### Smoothing Pipeline (critical for scoring)

```python
def scoring_aware_smooth(q, marginal, mix_alpha=0.02, floor=0.003):
    # Stage 1: Bayesian shrinkage toward marginal
    q = (1 - mix_alpha) * q + mix_alpha * marginal
    # Stage 2: hard safety floor
    q = np.maximum(q, floor)
    return q / q.sum(axis=-1, keepdims=True)
```

**Key insights:**
- The marginal distribution (average across all training data per initial class) is a far better fallback than uniform
- Mixture smoothing with marginal is more principled than a flat floor — redistributes mass proportional to class frequency
- Floor of 0.003 is near-optimal: steals only 1.8% of total mass vs 6% for 0.01
- Tune (mix_alpha, floor) on training data

### Spatial Smoothing

Nearby cells have correlated outcomes. Apply Gaussian smoothing per class channel, then renormalize:

```python
from scipy.ndimage import gaussian_filter

def spatial_smooth(prob_grid, sigma=1.0):
    smoothed = np.stack([gaussian_filter(prob_grid[:,:,k], sigma) for k in range(6)], axis=-1)
    smoothed = np.clip(smoothed, 1e-8, None)
    return smoothed / smoothed.sum(axis=-1, keepdims=True)
```

Skip smoothing for deterministic cells (mountains, deep ocean). Tune σ ∈ [0.3, 0.5, 1.0, 1.5, 2.0] on training data.

### Dirichlet-Multinomial Fusion (if using API observations)

If we have simulator/model predictions + API observation samples, fuse with conjugate Bayesian update:

```python
def fused_estimate(model_probs, observation_counts, prior_weight=20, jeffreys=0.5):
    alpha = prior_weight * model_probs + jeffreys
    posterior = alpha + observation_counts
    return posterior / posterior.sum()
```

`prior_weight` controls trust in model vs observations. Tune on training data.

## API Observation Strategy

50 queries = 50 independent stochastic samples. Each shows a 5-15 cell viewport from a different random simulation run. Observations give:
- Terrain grid (one sample from the distribution)
- Full settlement stats (population, food, wealth, defense, owner_id)

**Potential uses:**
- Aggregate multiple samples at same viewport to estimate local distribution
- Use settlement stats to infer round-specific hidden parameters (growth rate, aggression)
- Focus queries on high-entropy cells where our model is most uncertain

**Limitation:** 10 queries per seed × 225 cells = 2250 cell-samples, but each is a single draw from ~200. Sparse signal for distribution estimation.

## Implementation Priority

1. **Compute proper marginal distributions** from all 25 training pairs — zero-parameter strong baseline
2. **Fix smoothing** — switch from flat 0.01 floor to marginal mixture + 0.003 floor
3. **Add more features** to the lookup/model — neighborhood counts, settlement density
4. **Train LightGBM** on cell-level features with entropy-weighted cross-entropy loss
5. **Tune all hyperparameters** on leave-one-round-out CV using actual competition score formula
6. **Spatial smoothing** — Gaussian filter with tuned sigma
7. **Ensemble** predictions from multiple approaches