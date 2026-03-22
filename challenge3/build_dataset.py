"""Pull all ground truth data from completed rounds and save as training dataset."""

import numpy as np
import json
from pathlib import Path
from challenge3.api import get_rounds, get_round, get_analysis

GRID_VALUE_TO_CLASS = {10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
DATA_DIR = Path("challenge3/data")


def raw_grid_to_classes(grid_raw):
    """Convert raw grid values to prediction classes."""
    arr = np.array(grid_raw)
    return np.vectorize(lambda v: GRID_VALUE_TO_CLASS.get(v, 0))(arr)


def pull_all_training_data():
    DATA_DIR.mkdir(exist_ok=True)

    all_rounds = get_rounds()
    completed = [r for r in all_rounds if r["status"] == "completed"]
    print(f"Found {len(completed)} completed rounds")

    dataset = []

    for rnd in completed:
        rid = rnd["id"]
        rnum = rnd["round_number"]
        rd = get_round(rid)

        for seed in range(5):
            print(f"  Round {rnum} seed {seed}...", end=" ")

            try:
                analysis = get_analysis(rid, seed)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            gt = np.array(analysis["ground_truth"])
            init_grid_raw = np.array(analysis["initial_grid"])
            init_classes = raw_grid_to_classes(init_grid_raw)

            # Also get settlement positions from round data
            settlements = rd["initial_states"][seed]["settlements"]

            entry = {
                "round_id": rid,
                "round_number": rnum,
                "seed": seed,
                "initial_grid_raw": init_grid_raw,
                "initial_classes": init_classes,
                "ground_truth": gt,
                "settlements": settlements,
            }
            dataset.append(entry)

            # Quick stats
            gt_entropy = -(gt * np.log(gt + 1e-10)).sum(axis=-1)
            dynamic = (gt_entropy > 0.1).sum()
            print(f"OK (dynamic cells: {dynamic}/1600)")

    # Save as numpy archive
    out_path = DATA_DIR / "training_data.npz"
    np.savez_compressed(
        out_path,
        # Store arrays
        initial_grids_raw=np.array([d["initial_grid_raw"] for d in dataset]),
        initial_classes=np.array([d["initial_classes"] for d in dataset]),
        ground_truths=np.array([d["ground_truth"] for d in dataset]),
        # Store metadata as JSON string
        metadata=json.dumps([
            {
                "round_id": d["round_id"],
                "round_number": d["round_number"],
                "seed": d["seed"],
                "settlements": d["settlements"],
            }
            for d in dataset
        ]),
    )
    print(f"\nSaved {len(dataset)} training examples to {out_path}")
    print(f"  initial_grids_raw: {np.array([d['initial_grid_raw'] for d in dataset]).shape}")
    print(f"  initial_classes: {np.array([d['initial_classes'] for d in dataset]).shape}")
    print(f"  ground_truths: {np.array([d['ground_truth'] for d in dataset]).shape}")

    return dataset


def load_dataset():
    """Load the saved training dataset."""
    data = np.load(DATA_DIR / "training_data.npz", allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    return {
        "initial_grids_raw": data["initial_grids_raw"],
        "initial_classes": data["initial_classes"],
        "ground_truths": data["ground_truths"],
        "metadata": metadata,
    }


if __name__ == "__main__":
    pull_all_training_data()