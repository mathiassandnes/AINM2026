"""Clean solver: maximum regime detection → LightGBM predict → submit.

Key insight: Dirichlet updates with 1 observation per cell HURT scores (-12 to -20 pts).
All 50 queries are better spent on diverse regime probes across multiple seeds/viewports
to get the most accurate round-level feature estimates possible.
"""

import numpy as np
import sys
from pathlib import Path
from challenge3.api import (
    get_round, get_rounds, simulate, submit, get_budget, get_my_rounds,
)
from challenge3.build_dataset import load_dataset, raw_grid_to_classes
from challenge3.model import (
    build_feature_matrix, train_models, predict_models,
    extract_cell_features, apply_hard_rules, smooth,
    build_marginals, compute_score, estimate_round_features_from_observations,
    ROUND_FEATURE_NAMES, ALL_FEATURE_NAMES,
    MAP_SIZE, NUM_CLASSES,
)

GRID_VALUE_TO_CLASS_OBS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


# === Viewport helpers ===

def find_settlement_viewport(ic):
    """Find 15x15 viewport with most initial settlements."""
    best_x, best_y, best_count = 0, 0, 0
    for vy in range(MAP_SIZE - 14):
        for vx in range(MAP_SIZE - 14):
            count = ((ic[vy:vy + 15, vx:vx + 15] == 1) | (ic[vy:vy + 15, vx:vx + 15] == 2)).sum()
            if count > best_count:
                best_count = count
                best_x, best_y = vx, vy
    return best_x, best_y, best_count


def find_empty_viewport(ic):
    """Find 15x15 viewport with fewest settlements."""
    best_x, best_y, best_count = 0, 0, 9999
    for vy in range(0, MAP_SIZE - 14, 5):
        for vx in range(0, MAP_SIZE - 14, 5):
            count = ((ic[vy:vy + 15, vx:vx + 15] == 1) | (ic[vy:vy + 15, vx:vx + 15] == 2)).sum()
            if count < best_count:
                best_count = count
                best_x, best_y = vx, vy
    return best_x, best_y


def find_coastal_viewport(ic, init_grid_raw):
    """Find 15x15 viewport near coast (mix of ocean and land)."""
    ocean_mask = (init_grid_raw == 10)
    best_x, best_y, best_score = 0, 0, -1
    for vy in range(0, MAP_SIZE - 14, 5):
        for vx in range(0, MAP_SIZE - 14, 5):
            ocean_frac = ocean_mask[vy:vy + 15, vx:vx + 15].mean()
            score = min(ocean_frac, 1 - ocean_frac)
            if score > best_score:
                best_score = score
                best_x, best_y = vx, vy
    return best_x, best_y


def tiled_viewports(n_viewports):
    """Generate non-overlapping tile positions covering the map evenly.

    9 tiles cover the full 40x40 map. If n_viewports > 9, wraps around.
    This gives maximum spatial coverage for estimating expansion patterns.
    """
    tiles = []
    for vy in range(0, MAP_SIZE, 15):
        for vx in range(0, MAP_SIZE, 15):
            tiles.append((vx, vy, "tile"))
    # If we need more than 9, repeat (different stochastic sample each time)
    result = []
    for i in range(n_viewports):
        vx, vy, label = tiles[i % len(tiles)]
        result.append((vx, vy, label))
    return result


# === Observation ===

def probe_viewport(round_id, seed_index, vx, vy, w=15, h=15):
    """Single observation query. Returns parsed result."""
    result = simulate(round_id, seed_index, vx, vy, w, h)
    grid = np.array(result["grid"])
    observed = np.vectorize(lambda v: GRID_VALUE_TO_CLASS_OBS.get(v, 0))(grid)
    viewport = result.get("viewport", {"x": vx, "y": vy, "w": w, "h": h})
    return {
        "observed": observed,
        "settlements": result.get("settlements", []),
        "viewport": viewport,
        "seed_index": seed_index,
    }


# === Main solver ===

class Solver:
    def __init__(self, round_id=None):
        if round_id is None:
            for r in get_rounds():
                if r["status"] == "active":
                    round_id = r["id"]
                    break
            if round_id is None:
                raise ValueError("No active round")

        self.round_id = round_id
        self.rd = get_round(round_id)
        self.n_seeds = self.rd["seeds_count"]
        self.round_number = self.rd["round_number"]
        self.round_weight = self.rd["round_weight"]

        # Load training data
        self.dataset = load_dataset()
        self.global_marginal, self.class_marginals = build_marginals(
            self.dataset["ground_truths"], self.dataset["initial_classes"]
        )

        # Pre-compute initial class maps
        self.init_grids = [
            np.array(self.rd["initial_states"][s]["grid"]) for s in range(self.n_seeds)
        ]
        self.init_classes = [
            raw_grid_to_classes(g) for g in self.init_grids
        ]

        # State
        self.probes = []
        self.models = None
        self.round_features_est = None

        budget = get_budget()
        print(f"Round {self.round_number} (weight {self.round_weight})")
        print(f"Budget: {budget['queries_used']}/{budget['queries_max']}")
        print(f"Training: {len(self.dataset['metadata'])} examples")

    def probe_all(self):
        """Use ALL available budget for tiled regime detection probes.

        Strategy: tile the map systematically across seeds for maximum spatial
        coverage. 9 tiles cover the full 40x40 map per seed, so with 50 queries
        we get ~5 seeds × 9-10 tiles = full map coverage on most seeds.
        This gives the best expansion_rate estimate since we directly measure
        P(settlement | initially_empty) across the whole map.
        """
        budget = get_budget()
        remaining = budget["queries_max"] - budget["queries_used"]
        print(f"\n=== Regime detection ({remaining} queries available) ===")

        # Distribute queries evenly across seeds
        probes_per_seed = remaining // self.n_seeds
        leftover = remaining - probes_per_seed * self.n_seeds

        for seed in range(self.n_seeds):
            n = probes_per_seed + (1 if seed < leftover else 0)
            viewports = tiled_viewports(n)

            for vx, vy, vtype in viewports:
                w = min(15, MAP_SIZE - vx)
                h = min(15, MAP_SIZE - vy)
                probe = probe_viewport(self.round_id, seed, vx, vy, w, h)
                self.probes.append(probe)
                n_s = (probe["observed"] == 1).sum() + (probe["observed"] == 2).sum()
                n_r = (probe["observed"] == 3).sum()
                alive = [s for s in probe["settlements"] if s.get("alive")]
                print(f"  Seed {seed} ({vtype:>11}): {n_s:3d} settle, {n_r:2d} ruins, "
                      f"{len(alive):2d} alive")

        # Estimate round-level features from all probes
        all_grids = {s: self.init_grids[s] for s in range(self.n_seeds)}
        self.round_features_est = estimate_round_features_from_observations(
            self.probes, all_grids,
        )
        print(f"\n  Estimated round features ({len(self.probes)} probes):")
        for k, v in self.round_features_est.items():
            print(f"    {k}: {v:.4f}")

    def predict_and_submit(self, mix_alpha=0.05, floor=0.001, sigma=0.0):
        """Train model on all training data and submit predictions for all seeds."""
        if self.round_features_est is None:
            raise ValueError("Run probe_all first")

        print(f"\n=== Train model & submit ===")

        # Build feature matrix and train
        X, Y, _, _ = build_feature_matrix(self.dataset)
        print(f"  Training on {X.shape[0]} cell examples")
        self.models = train_models(X, Y, feature_names=ALL_FEATURE_NAMES)
        print(f"  Trained 6 LightGBM models")

        names = ["Empty", "Settle", "Port", "Ruin", "Forest", "Mount"]

        for seed in range(self.n_seeds):
            cell_feats = extract_cell_features(self.init_grids[seed], self.init_classes[seed])
            rf_vec = np.array(
                [self.round_features_est[k] for k in ROUND_FEATURE_NAMES],
                dtype=np.float32,
            )
            rf_tiled = np.tile(rf_vec, (MAP_SIZE * MAP_SIZE, 1))
            X_seed = np.hstack([cell_feats, rf_tiled])

            pred_flat = predict_models(self.models, X_seed)
            pred = pred_flat.reshape(MAP_SIZE, MAP_SIZE, NUM_CLASSES)
            pred = apply_hard_rules(pred, self.init_grids[seed], self.init_classes[seed])
            pred = smooth(pred, self.global_marginal, mix_alpha, floor, sigma)

            argmax = pred.argmax(axis=-1)
            print(f"\n  Seed {seed}:")
            for cls in range(NUM_CLASSES):
                n = (argmax == cls).sum()
                print(f"    {names[cls]:>7}: {n:4d} argmax cells")

            print(f"  Submitting seed {seed}...")
            try:
                result = submit(self.round_id, seed, pred.tolist())
                print(f"    {result}")
            except Exception as e:
                print(f"    Failed: {e}")

    def status(self):
        """Print current status."""
        budget = get_budget()
        print(f"\nBudget: {budget['queries_used']}/{budget['queries_max']}")
        for r in get_my_rounds():
            if r.get("id") == self.round_id:
                print(f"Submitted: {r.get('seeds_submitted', '?')}/5")
                if r.get("seed_scores"):
                    print(f"Scores: {r['seed_scores']}")
                break

    def run(self):
        """Full pipeline: probe all → predict → submit."""
        self.probe_all()
        self.predict_and_submit()
        self.status()

    def dry_run(self, n_probes=10):
        """Probe and predict but don't submit."""
        print("[DRY RUN — limited probes, no submissions]")

        # Use only n_probes for dry run to conserve budget
        budget = get_budget()
        remaining = budget["queries_max"] - budget["queries_used"]
        actual_probes = min(n_probes, remaining)

        seeds_to_probe = list(range(min(3, self.n_seeds)))
        probes_per_seed = max(1, actual_probes // len(seeds_to_probe))

        for seed in seeds_to_probe:
            ic = self.init_classes[seed]
            raw = self.init_grids[seed]
            viewports = diverse_viewports(ic, raw, probes_per_seed)

            for vx, vy, vtype in viewports:
                probe = probe_viewport(self.round_id, seed, vx, vy)
                self.probes.append(probe)
                n_s = (probe["observed"] == 1).sum() + (probe["observed"] == 2).sum()
                n_r = (probe["observed"] == 3).sum()
                print(f"  Seed {seed} ({vtype}): {n_s} settle, {n_r} ruins")

        all_grids = {s: self.init_grids[s] for s in range(self.n_seeds)}
        self.round_features_est = estimate_round_features_from_observations(
            self.probes, all_grids,
        )
        print(f"\n  Estimated round features ({len(self.probes)} probes):")
        for k, v in self.round_features_est.items():
            print(f"    {k}: {v:.4f}")

        X, Y, _, _ = build_feature_matrix(self.dataset)
        self.models = train_models(X, Y, feature_names=ALL_FEATURE_NAMES)

        names = ["Empty", "Settle", "Port", "Ruin", "Forest", "Mount"]
        for seed in range(self.n_seeds):
            cell_feats = extract_cell_features(self.init_grids[seed], self.init_classes[seed])
            rf_vec = np.array(
                [self.round_features_est[k] for k in ROUND_FEATURE_NAMES],
                dtype=np.float32,
            )
            rf_tiled = np.tile(rf_vec, (MAP_SIZE * MAP_SIZE, 1))
            X_seed = np.hstack([cell_feats, rf_tiled])

            pred_flat = predict_models(self.models, X_seed)
            pred = pred_flat.reshape(MAP_SIZE, MAP_SIZE, NUM_CLASSES)
            pred = apply_hard_rules(pred, self.init_grids[seed], self.init_classes[seed])
            pred = smooth(pred, self.global_marginal)

            argmax = pred.argmax(axis=-1)
            print(f"\n  Seed {seed}:")
            for cls in range(NUM_CLASSES):
                n = (argmax == cls).sum()
                avg_p = pred[:, :, cls].mean()
                print(f"    {names[cls]:>7}: {n:4d} argmax, avg P={avg_p:.4f}")

        print("\n[DRY RUN COMPLETE — nothing submitted]")
        self.status()


def main():
    dry = "--dry" in sys.argv
    rid = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            rid = arg

    solver = Solver(rid)
    if dry:
        solver.dry_run()
    else:
        solver.run()


if __name__ == "__main__":
    main()
