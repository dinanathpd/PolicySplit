"""
prepare_datasets.py — Load CIC-IoT-2023 CSVs, group labels, split, and save.

Usage:
    python scripts/prepare_datasets.py --config config/cnn_config.yaml

Output:
    data/train.npz
    data/val.npz
    data/test.npz
    data/class_weights.npy
    data/train_manifest.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.preprocessing.feature_extractor import (  # noqa: E402
    FeatureExtractor,
    GROUP_LABEL_TO_ID,
    ID_TO_GROUP_LABEL,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CIC-IoT-2023 dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_cic_iot_2023/CSV/MERGED_CSV",
        help="Path to MERGED_CSV directory",
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--num-classes", type=int, default=8, choices=[8, 34])
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of CSV files to load (for fast dev)")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of rows to sample from each CSV")
    parser.add_argument("--balanced", action="store_true",
                        help="Balance classes by downsampling majority to "
                             "match minority (+ optional oversample)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="When --balanced, cap each class at this many "
                             "samples. Default: match smallest class.")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Target total dataset size. Minority classes keep "
                             "ALL their real samples; majority classes share "
                             "the remaining budget equally.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load ---
    extractor = FeatureExtractor(
        args.dataset_dir, num_classes=args.num_classes)
    X, y = extractor.load_all(max_files=args.max_files,
                              sample_frac=args.sample_frac)

    logger.info("Dataset shape: X=%s  y=%s", X.shape, y.shape)

    # --- Balance classes (optional) ---
    if args.total_samples:
        # Mode: keep ALL minority real samples; majority fills the rest
        rng = np.random.RandomState(args.seed)
        classes = np.unique(y)
        n_classes = len(classes)
        class_counts = {c: int((y == c).sum()) for c in classes}

        fair_share = args.total_samples // n_classes
        logger.info("Total-samples mode: %d total, fair share = %d/class",
                    args.total_samples, fair_share)

        # Iteratively allocate: small classes keep all, surplus goes to rest
        allocation = {}
        remaining_budget = args.total_samples
        remaining_classes = set(classes)

        # Sort by count ascending so minorities are handled first
        for c in sorted(classes, key=lambda c: class_counts[c]):
            share = remaining_budget // len(remaining_classes)
            real = class_counts[c]
            if real <= share:
                # Minority: keep all real samples (no oversampling)
                allocation[c] = real
            else:
                # Majority: downsample to share
                allocation[c] = share
            remaining_budget -= allocation[c]
            remaining_classes.remove(c)

        balanced_idx = []
        for c in classes:
            idx = np.where(y == c)[0]
            target = allocation[c]
            if len(idx) >= target:
                chosen = rng.choice(idx, size=target, replace=False)
            else:
                chosen = idx  # keep all (shouldn't oversample)
            balanced_idx.append(chosen)

        balanced_idx = np.concatenate(balanced_idx)
        rng.shuffle(balanced_idx)
        X = X[balanced_idx]
        y = y[balanced_idx]

        logger.info("Total-samples allocation (%d actual):", len(y))
        for c in sorted(classes):
            name = ID_TO_GROUP_LABEL.get(c, str(c))
            logger.info("  %s: %d (real: %d)", name, allocation[c],
                        class_counts[c])

    elif args.balanced:
        rng = np.random.RandomState(args.seed)
        classes = np.unique(y)
        class_counts = {c: (y == c).sum() for c in classes}
        min_count = min(class_counts.values())

        if args.max_per_class:
            target = args.max_per_class
        else:
            target = min_count

        logger.info("Balancing: target %d samples per class (min class has %d)",
                    target, min_count)

        balanced_idx = []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) >= target:
                # Downsample
                chosen = rng.choice(idx, size=target, replace=False)
            else:
                # Oversample (repeat with replacement to reach target)
                chosen = rng.choice(idx, size=target, replace=True)
            balanced_idx.append(chosen)

        balanced_idx = np.concatenate(balanced_idx)
        rng.shuffle(balanced_idx)
        X = X[balanced_idx]
        y = y[balanced_idx]
        logger.info("Balanced dataset: %d samples (%d per class × %d classes)",
                    len(y), target, len(classes))

    # --- Class distribution ---
    dist = extractor.class_distribution(y)
    label_names = ID_TO_GROUP_LABEL if args.num_classes == 8 else None
    logger.info("Class distribution:")
    for cls_id, count in sorted(dist.items()):
        name = label_names.get(cls_id, str(
            cls_id)) if label_names else str(cls_id)
        logger.info("  %s (%d): %d samples", name, cls_id, count)

    # --- Split ---
    splits = FeatureExtractor.stratified_split(
        X, y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, (X_s, y_s) in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.npz")
        np.savez_compressed(path, X=X_s, y=y_s)
        logger.info("Saved %s: %d samples → %s", split_name, len(y_s), path)

    # --- Class weights ---
    weights = FeatureExtractor.compute_class_weights(splits["train"][1])
    np.save(os.path.join(args.output_dir, "class_weights.npy"), weights)
    logger.info("Class weights: %s", weights)

    # --- Manifest ---
    manifest = {
        "num_classes": args.num_classes,
        "num_features": int(X.shape[1]),
        "train_samples": int(len(splits["train"][1])),
        "val_samples": int(len(splits["val"][1])),
        "test_samples": int(len(splits["test"][1])),
        "class_distribution": {str(k): v for k, v in dist.items()},
        "class_names": (
            {str(v): k for k, v in GROUP_LABEL_TO_ID.items()}
            if args.num_classes == 8
            else None
        ),
    }
    manifest_path = os.path.join(args.output_dir, "train_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
