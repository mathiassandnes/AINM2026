"""Leave-one-round-out CV evaluation with correct scoring formula."""

import numpy as np
from itertools import product
from challenge3.build_dataset import load_dataset
from challenge3.model import (
    build_feature_matrix, train_models, predict_models,
    apply_hard_rules, smooth, compute_score, build_marginals,
    extract_cell_features, ROUND_FEATURE_NAMES, ALL_FEATURE_NAMES,
    MAP_SIZE, NUM_CLASSES,
)


def evaluate_lgbm(dataset, tune=False):
    """Leave-one-round-out CV for LightGBM model."""
    init_grids_raw = dataset["initial_grids_raw"]
    init_classes = dataset["initial_classes"]
    ground_truths = dataset["ground_truths"]
    metadata = dataset["metadata"]

    n = len(metadata)
    round_numbers = [m["round_number"] for m in metadata]
    unique_rounds = sorted(set(round_numbers))

    # Build full feature matrix
    X, Y, example_round_numbers, round_features = build_feature_matrix(dataset)

    # Global marginals (from all data, for smoothing baseline)
    global_marginal_all, _ = build_marginals(ground_truths, init_classes)

    print(f"Dataset: {n} examples across rounds {unique_rounds}")
    print(f"Feature matrix: {X.shape}, Target: {Y.shape}")
    print(f"Round features: {round_features}")
    print()

    all_scores = {}

    for test_round in unique_rounds:
        train_idx = [i for i in range(n) if round_numbers[i] != test_round]
        test_idx = [i for i in range(n) if round_numbers[i] == test_round]

        # Build train/test split at cell level
        train_cell_idx = []
        for i in train_idx:
            start = i * MAP_SIZE * MAP_SIZE
            train_cell_idx.extend(range(start, start + MAP_SIZE * MAP_SIZE))

        X_train = X[train_cell_idx]
        Y_train = Y[train_cell_idx]

        # Train marginals from training data only
        global_marginal, _ = build_marginals(
            ground_truths[train_idx], init_classes[train_idx]
        )

        # Train LightGBM
        models = train_models(X_train, Y_train, feature_names=ALL_FEATURE_NAMES)

        # Evaluate on test round
        round_scores = []
        for i in test_idx:
            start = i * MAP_SIZE * MAP_SIZE
            end = start + MAP_SIZE * MAP_SIZE
            X_test = X[start:end]
            gt = ground_truths[i]

            pred_flat = predict_models(models, X_test)
            pred = pred_flat.reshape(MAP_SIZE, MAP_SIZE, NUM_CLASSES)
            pred = apply_hard_rules(pred, init_grids_raw[i], init_classes[i])
            pred = smooth(pred, global_marginal)

            score = compute_score(pred, gt)
            round_scores.append(score)

        avg = np.mean(round_scores)
        all_scores[test_round] = avg
        print(f"Round {test_round}: {avg:.2f} "
              f"(seeds: {', '.join(f'{s:.1f}' for s in round_scores)})")

    overall = np.mean(list(all_scores.values()))
    print(f"\n=== LightGBM Overall CV: {overall:.2f} ===")

    if tune:
        print("\n--- Tuning smoothing params ---")
        best_score = overall
        best_params = (0.02, 0.003, 0.5)

        for alpha, floor, sigma in product(
            [0.01, 0.02, 0.03, 0.05],
            [0.001, 0.003, 0.005],
            [0.0, 0.3, 0.5, 0.7, 1.0],
        ):
            cv_scores = []
            for test_round in unique_rounds:
                train_idx = [i for i in range(n) if round_numbers[i] != test_round]
                test_idx = [i for i in range(n) if round_numbers[i] == test_round]

                train_cell_idx = []
                for i in train_idx:
                    start = i * MAP_SIZE * MAP_SIZE
                    train_cell_idx.extend(range(start, start + MAP_SIZE * MAP_SIZE))

                global_marginal, _ = build_marginals(
                    ground_truths[train_idx], init_classes[train_idx]
                )
                models = train_models(X[train_cell_idx], Y[train_cell_idx],
                                      feature_names=ALL_FEATURE_NAMES)

                for i in test_idx:
                    start = i * MAP_SIZE * MAP_SIZE
                    pred_flat = predict_models(models, X[start:start + MAP_SIZE * MAP_SIZE])
                    pred = pred_flat.reshape(MAP_SIZE, MAP_SIZE, NUM_CLASSES)
                    pred = apply_hard_rules(pred, init_grids_raw[i], init_classes[i])
                    pred = smooth(pred, global_marginal, alpha, floor, sigma)
                    cv_scores.append(compute_score(pred, ground_truths[i]))

            avg = np.mean(cv_scores)
            if avg > best_score:
                best_score = avg
                best_params = (alpha, floor, sigma)
                print(f"  New best: alpha={alpha}, floor={floor}, sigma={sigma} → {avg:.2f}")

        print(f"\nBest params: alpha={best_params[0]}, floor={best_params[1]}, sigma={best_params[2]} → {best_score:.2f}")

    return all_scores


def evaluate_lookup_baseline(dataset):
    """Baseline: lookup table for comparison."""
    from collections import defaultdict
    from challenge3.model import min_manhattan_dist

    init_grids_raw = dataset["initial_grids_raw"]
    init_classes = dataset["initial_classes"]
    ground_truths = dataset["ground_truths"]
    metadata = dataset["metadata"]

    n = len(metadata)
    round_numbers = [m["round_number"] for m in metadata]
    unique_rounds = sorted(set(round_numbers))

    all_scores = {}

    for test_round in unique_rounds:
        train_idx = [i for i in range(n) if round_numbers[i] != test_round]
        test_idx = [i for i in range(n) if round_numbers[i] == test_round]

        global_marginal, _ = build_marginals(
            ground_truths[train_idx], init_classes[train_idx]
        )

        # Build 3-feature lookup
        lookup = defaultdict(list)
        for i in train_idx:
            ic_map = init_classes[i]
            raw_map = init_grids_raw[i]
            gt_map = ground_truths[i]
            settle_mask = (ic_map == 1) | (ic_map == 2)
            ocean_mask = (raw_map == 10)
            ds = min_manhattan_dist(settle_mask)
            do = min_manhattan_dist(ocean_mask)
            for y in range(MAP_SIZE):
                for x in range(MAP_SIZE):
                    key = (int(ic_map[y, x]), min(int(ds[y, x]), 12),
                           1 if do[y, x] <= 1 else 0)
                    lookup[key].append(gt_map[y, x])
        lookup_avg = {k: np.mean(v, axis=0) for k, v in lookup.items()}

        round_scores = []
        for i in test_idx:
            ic = init_classes[i]
            raw = init_grids_raw[i]
            gt = ground_truths[i]
            settle_mask = (ic == 1) | (ic == 2)
            ocean_mask = (raw == 10)
            ds = min_manhattan_dist(settle_mask)
            do = min_manhattan_dist(ocean_mask)

            pred = np.zeros((MAP_SIZE, MAP_SIZE, NUM_CLASSES))
            for y in range(MAP_SIZE):
                for x in range(MAP_SIZE):
                    key = (int(ic[y, x]), min(int(ds[y, x]), 12),
                           1 if do[y, x] <= 1 else 0)
                    pred[y, x] = lookup_avg.get(key, global_marginal)

            pred = smooth(pred, global_marginal)
            round_scores.append(compute_score(pred, gt))

        avg = np.mean(round_scores)
        all_scores[test_round] = avg
        print(f"Round {test_round}: {avg:.2f} "
              f"(seeds: {', '.join(f'{s:.1f}' for s in round_scores)})")

    overall = np.mean(list(all_scores.values()))
    print(f"\n=== Lookup Baseline Overall CV: {overall:.2f} ===")
    return all_scores


def run_evaluation():
    dataset = load_dataset()

    print("=" * 60)
    print("LOOKUP TABLE BASELINE (correct x3 scoring)")
    print("=" * 60)
    lookup_scores = evaluate_lookup_baseline(dataset)

    print()
    print("=" * 60)
    print("LIGHTGBM MODEL (correct x3 scoring)")
    print("=" * 60)
    lgbm_scores = evaluate_lgbm(dataset)

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    rounds = sorted(set(lookup_scores.keys()) | set(lgbm_scores.keys()))
    print(f"{'Round':>6} {'Lookup':>8} {'LightGBM':>10} {'Diff':>8}")
    for r in rounds:
        l = lookup_scores.get(r, 0)
        g = lgbm_scores.get(r, 0)
        print(f"{r:>6} {l:>8.2f} {g:>10.2f} {g - l:>+8.2f}")
    print(f"{'Avg':>6} {np.mean(list(lookup_scores.values())):>8.2f} "
          f"{np.mean(list(lgbm_scores.values())):>10.2f} "
          f"{np.mean(list(lgbm_scores.values())) - np.mean(list(lookup_scores.values())):>+8.2f}")


if __name__ == "__main__":
    import sys
    tune = "--tune" in sys.argv
    if tune:
        dataset = load_dataset()
        evaluate_lgbm(dataset, tune=True)
    else:
        run_evaluation()
