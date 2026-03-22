"""Feature extraction + LightGBM model for Astar Island prediction."""

import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter, uniform_filter
import lightgbm as lgb

MAP_SIZE = 40
NUM_CLASSES = 6


# === Feature extraction ===

def min_manhattan_dist(mask):
    """BFS manhattan distance from mask positions."""
    dist = np.full((MAP_SIZE, MAP_SIZE), 99, dtype=int)
    q = deque()
    for y, x in zip(*np.where(mask)):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE and dist[ny, nx] > dist[cy, cx] + 1:
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((ny, nx))
    return dist


def extract_cell_features(init_grid_raw, init_classes):
    """Extract per-cell features from initial state. Returns (1600, n_features) array."""
    ic = init_classes
    settle_mask = (ic == 1) | (ic == 2)
    ocean_mask = (init_grid_raw == 10)
    mountain_mask = (ic == 5)
    forest_mask = (ic == 4)

    dist_settle = min_manhattan_dist(settle_mask)
    dist_ocean = min_manhattan_dist(ocean_mask)
    dist_mountain = min_manhattan_dist(mountain_mask)
    dist_forest = min_manhattan_dist(forest_mask)

    # Settlement counts in radius 3 and 5
    settle_float = settle_mask.astype(float)
    settle_count_r3 = uniform_filter(settle_float, size=7, mode='constant') * 49  # 7x7 window
    settle_count_r5 = uniform_filter(settle_float, size=11, mode='constant') * 121  # 11x11 window

    # Forest fraction in radius 3
    forest_float = forest_mask.astype(float)
    forest_frac_r3 = uniform_filter(forest_float, size=7, mode='constant')

    features = []
    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            cell_ic = int(ic[y, x])
            # One-hot initial class
            one_hot = [0.0] * NUM_CLASSES
            one_hot[cell_ic] = 1.0

            feat = one_hot + [
                min(int(dist_settle[y, x]), 15),    # dist_to_settlement (capped)
                min(int(dist_ocean[y, x]), 15),      # dist_to_ocean
                min(int(dist_mountain[y, x]), 15),   # dist_to_mountain
                min(int(dist_forest[y, x]), 15),     # dist_to_forest
                settle_count_r3[y, x],               # settlement_count_r3
                settle_count_r5[y, x],               # settlement_count_r5
                forest_frac_r3[y, x],                # forest_fraction_r3
                1.0 if dist_ocean[y, x] <= 1 else 0.0,  # adj_ocean
                1.0 if init_grid_raw[y, x] == 10 else 0.0,  # is_ocean
                1.0 if ic[y, x] == 5 else 0.0,      # is_mountain
            ]
            features.append(feat)
    return np.array(features, dtype=np.float32)


CELL_FEATURE_NAMES = [
    "ic_0", "ic_1", "ic_2", "ic_3", "ic_4", "ic_5",
    "dist_settle", "dist_ocean", "dist_mountain", "dist_forest",
    "settle_count_r3", "settle_count_r5", "forest_frac_r3",
    "adj_ocean", "is_ocean", "is_mountain",
]


def compute_round_features_from_gt(ground_truths, init_classes):
    """Compute round-level features from ground truth distributions.

    Args:
        ground_truths: (n_seeds, 40, 40, 6) ground truth for seeds of one round
        init_classes: (n_seeds, 40, 40) initial class maps
    Returns:
        dict of round-level features
    """
    expansions = []
    ruin_rates = []
    for i in range(len(ground_truths)):
        gt_i = ground_truths[i]
        ic_i = init_classes[i]
        empty_mask = ic_i == 0
        if empty_mask.sum() > 0:
            expansions.append(gt_i[empty_mask][:, 1].mean())
        ruin_rates.append(gt_i[:, :, 3].mean())

    expansion_rate = np.mean(expansions) if expansions else 0.0

    # Estimate expansion reach: distance where P(settle) drops below threshold
    # Use all seeds, compute P(settle) vs distance to settlement
    all_dists = []
    all_settle_probs = []
    for i in range(len(ground_truths)):
        ic_i = init_classes[i]
        settle_mask = (ic_i == 1) | (ic_i == 2)
        dist = min_manhattan_dist(settle_mask)
        empty_mask = ic_i == 0
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                if empty_mask[y, x]:
                    all_dists.append(dist[y, x])
                    all_settle_probs.append(ground_truths[i][y, x, 1])
    all_dists = np.array(all_dists)
    all_settle_probs = np.array(all_settle_probs)

    # Expansion reach: max distance where mean P(settle) > 0.5 * expansion_rate
    threshold = 0.5 * max(expansion_rate, 0.001)
    expansion_reach = 0
    for d in range(1, 20):
        mask = all_dists == d
        if mask.sum() > 0 and all_settle_probs[mask].mean() > threshold:
            expansion_reach = d

    return {
        "expansion_rate": expansion_rate,
        "expansion_reach": float(expansion_reach),
        "ruin_rate": np.mean(ruin_rates),
        "avg_food": 0.0,  # not available from GT
        "avg_defense": 0.0,  # not available from GT
    }


ROUND_FEATURE_NAMES = [
    "expansion_rate", "expansion_reach", "ruin_rate", "avg_food", "avg_defense",
]


def build_feature_matrix(dataset):
    """Build full feature matrix from training dataset.

    Returns:
        X: (n_examples * 1600, n_cell_features + n_round_features)
        Y: (n_examples * 1600, 6)
        round_numbers: list of round numbers per example
    """
    init_grids_raw = dataset["initial_grids_raw"]
    init_classes = dataset["initial_classes"]
    ground_truths = dataset["ground_truths"]
    metadata = dataset["metadata"]

    n = len(metadata)

    # Compute round-level features per round
    round_nums = sorted(set(m["round_number"] for m in metadata))
    round_features = {}
    for rnum in round_nums:
        idxs = [i for i, m in enumerate(metadata) if m["round_number"] == rnum]
        round_features[rnum] = compute_round_features_from_gt(
            ground_truths[idxs], init_classes[idxs]
        )

    all_X = []
    all_Y = []
    example_round_numbers = []

    for i in range(n):
        rnum = metadata[i]["round_number"]
        cell_feats = extract_cell_features(init_grids_raw[i], init_classes[i])
        rf = round_features[rnum]
        round_feat_vec = np.array([rf[k] for k in ROUND_FEATURE_NAMES], dtype=np.float32)
        # Broadcast round features to all cells
        round_feats_tiled = np.tile(round_feat_vec, (MAP_SIZE * MAP_SIZE, 1))
        X_i = np.hstack([cell_feats, round_feats_tiled])
        Y_i = ground_truths[i].reshape(-1, NUM_CLASSES)

        all_X.append(X_i)
        all_Y.append(Y_i)
        example_round_numbers.append(rnum)

    X = np.vstack(all_X)
    Y = np.vstack(all_Y)
    return X, Y, example_round_numbers, round_features


ALL_FEATURE_NAMES = CELL_FEATURE_NAMES + ROUND_FEATURE_NAMES


# === LightGBM model ===

def train_models(X, Y, feature_names=None):
    """Train 6 LightGBM regressors, one per class."""
    models = []
    params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    for cls in range(NUM_CLASSES):
        ds = lgb.Dataset(X, label=Y[:, cls], feature_name=feature_names or "auto")
        model = lgb.train(params, ds, num_boost_round=200)
        models.append(model)
    return models


def predict_models(models, X):
    """Predict with 6 models and normalize to sum to 1."""
    preds = np.column_stack([m.predict(X) for m in models])
    preds = np.clip(preds, 1e-8, None)
    preds /= preds.sum(axis=1, keepdims=True)
    return preds


# === Hard rules ===

def apply_hard_rules(pred, init_grid_raw, init_classes):
    """Apply deterministic rules after model prediction.

    - Mountains → [0, 0, 0, 0, 0, 1]
    - Ocean (grid value 10) → [1, 0, 0, 0, 0, 0]
    - dist_ocean > 1 → port probability floored
    """
    pred = pred.copy()
    ocean_mask = init_grid_raw == 10
    mountain_mask = init_classes == 5
    dist_ocean = min_manhattan_dist(ocean_mask)

    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            if mountain_mask[y, x]:
                pred[y, x] = [0, 0, 0, 0, 0, 1]
            elif ocean_mask[y, x]:
                pred[y, x] = [1, 0, 0, 0, 0, 0]
            elif dist_ocean[y, x] > 1:
                # Ports only at ocean distance <= 1
                port_mass = pred[y, x, 2]
                if port_mass > 0.001:
                    pred[y, x, 2] = 0.001
                    # Redistribute mass proportionally to other classes
                    remaining = pred[y, x].copy()
                    remaining[2] = 0
                    if remaining.sum() > 0:
                        remaining = remaining / remaining.sum() * (1 - 0.001)
                        remaining[2] = 0.001
                        pred[y, x] = remaining
    return pred


# === Smoothing ===

def smooth(pred, global_marginal, mix_alpha=0.05, floor=0.001, sigma=0.0):
    """Scoring-aware smoothing: spatial Gaussian + marginal mixture + floor."""
    if sigma > 0:
        pred = np.stack(
            [gaussian_filter(pred[:, :, k], sigma) for k in range(NUM_CLASSES)],
            axis=-1,
        )
        pred = np.clip(pred, 1e-8, None)
        pred /= pred.sum(axis=-1, keepdims=True)

    pred = (1 - mix_alpha) * pred + mix_alpha * global_marginal
    pred = np.maximum(pred, floor)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


# === Scoring ===

def compute_score(pred, gt):
    """Competition score: 100 * exp(-3 * entropy_weighted_mean_KL)."""
    eps = 1e-10
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=-1)
    entropy = -np.sum(gt * np.log(gt + eps), axis=-1)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy
    return max(0.0, min(100.0, 100 * np.exp(-3 * weighted_kl)))


# === Full prediction pipeline ===

def predict_map(models, init_grid_raw, init_classes, round_level_features,
                global_marginal, mix_alpha=0.05, floor=0.001, sigma=0.0):
    """Full prediction pipeline: model → hard rules → smoothing."""
    cell_feats = extract_cell_features(init_grid_raw, init_classes)
    rf_vec = np.array([round_level_features[k] for k in ROUND_FEATURE_NAMES], dtype=np.float32)
    rf_tiled = np.tile(rf_vec, (MAP_SIZE * MAP_SIZE, 1))
    X = np.hstack([cell_feats, rf_tiled])

    pred_flat = predict_models(models, X)
    pred = pred_flat.reshape(MAP_SIZE, MAP_SIZE, NUM_CLASSES)

    pred = apply_hard_rules(pred, init_grid_raw, init_classes)
    pred = smooth(pred, global_marginal, mix_alpha, floor, sigma)
    return pred


# === Marginals (fallback/baseline) ===

def build_marginals(ground_truths, init_classes):
    """Build global and per-class marginal distributions."""
    global_marginal = ground_truths.mean(axis=(0, 1, 2))
    class_marginals = {}
    for cls in range(NUM_CLASSES):
        all_probs = []
        for i in range(len(ground_truths)):
            mask = init_classes[i] == cls
            if mask.sum() > 0:
                all_probs.append(ground_truths[i][mask])
        class_marginals[cls] = (
            np.concatenate(all_probs).mean(axis=0) if all_probs else global_marginal.copy()
        )
    return global_marginal, class_marginals


# === Observation-based round feature estimation ===

def estimate_round_features_from_observations(probes, initial_states_grids):
    """Estimate round-level features from simulation probes.

    Args:
        probes: list of dicts with keys: observed (class grid), settlements (list), viewport (dict)
        initial_states_grids: list of raw initial grids corresponding to each probe's seed
    Returns:
        dict of estimated round-level features
    """
    from challenge3.build_dataset import raw_grid_to_classes

    # Direct estimation: for each initially-empty cell in probed viewports,
    # check if it became a settlement. This directly measures expansion_rate
    # without needing a calibration formula.
    # Also track distance to nearest initial settlement for expansion_reach.
    total_empty = 0
    total_empty_became_settle = 0
    total_ruins = 0
    total_cells_observed = 0
    dist_settle_counts = {}  # distance -> (n_became_settle, n_total)

    for probe in probes:
        obs = probe["observed"]
        vp = probe["viewport"]
        raw_grid = np.array(initial_states_grids[probe["seed_index"]])
        ic = raw_grid_to_classes(raw_grid)
        settle_mask = (ic == 1) | (ic == 2)
        dist = min_manhattan_dist(settle_mask)

        vy, vx = vp["y"], vp["x"]
        vh, vw = vp["h"], vp["w"]
        viewport_ic = ic[vy:vy + vh, vx:vx + vw]
        viewport_dist = dist[vy:vy + vh, vx:vx + vw]

        total_cells_observed += obs.size
        total_ruins += (obs == 3).sum()

        for y in range(vh):
            for x in range(vw):
                if viewport_ic[y, x] == 0:  # initially empty
                    total_empty += 1
                    d = int(viewport_dist[y, x])
                    if d not in dist_settle_counts:
                        dist_settle_counts[d] = [0, 0]
                    dist_settle_counts[d][1] += 1
                    if obs[y, x] in (1, 2):  # became settlement
                        total_empty_became_settle += 1
                        dist_settle_counts[d][0] += 1

    # Direct expansion rate: fraction of initially-empty cells that became settlements
    # Apply 0.9x shrinkage to correct for single-observation upward bias
    expansion_rate = max(0.001, 0.9 * total_empty_became_settle / max(total_empty, 1))

    # Expansion reach: max distance where P(settle) > 0.02 with enough samples
    # Capped at 6 (max observed in all training rounds)
    expansion_reach = 1.0
    for d in sorted(dist_settle_counts.keys()):
        n_s, n_t = dist_settle_counts[d]
        if n_t >= 10 and n_s / n_t > 0.02:
            expansion_reach = float(min(d, 6))

    # Ruin rate: direct measurement
    ruin_fraction = total_ruins / max(total_cells_observed, 1)
    ruin_rate = ruin_fraction * 0.5

    return {
        "expansion_rate": expansion_rate,
        "expansion_reach": expansion_reach,
        "ruin_rate": ruin_rate,
        "avg_food": 0.0,
        "avg_defense": 0.0,
    }
