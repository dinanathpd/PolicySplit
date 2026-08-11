"""
train_cnn.py - Matched baseline suite.

Every baseline here is fitted on the same corpus, the same subsample, the
same partition and the same feature construction as the sequence models,
and each is evaluated under both context conditions. A comparison is only
informative if the compared systems differ in the estimator and in
nothing else, so no baseline is allowed its own preprocessing, its own
sample size or its own splitting rule.

Feature construction for the non-sequential estimators reduces each flow
to order statistics of its packet-length and inter-arrival series plus
its directional composition. This is the information a tabular model can
represent; it is derived from the same packets the sequence models see.

Usage:
    PYTHONPATH=. python scripts/train_cnn.py --split flow-isolated
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import protocol as P              # noqa: E402
from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def flow_summary(X_seq: np.ndarray) -> np.ndarray:
    """
    Order statistics of a flow's packet series.

    Columns: packet count, then the quantiles, mean and standard
    deviation of log packet length and of log inter-arrival time, then
    the share of packets travelling in the initiator's direction.
    """
    valid = X_seq[:, :, 0] != 0
    n = valid.sum(axis=1)
    cols = [n.astype(np.float32)]

    for c in (0, 1):
        chan = np.where(valid, X_seq[:, :, c], np.nan)
        with np.errstate(invalid="ignore"):
            cols.append(np.nanquantile(chan, QUANTILES, axis=1).T)
            cols.append(np.nanmean(chan, axis=1)[:, None])
            cols.append(np.nan_to_num(np.nanstd(chan, axis=1))[:, None])

    fwd = np.where(valid, X_seq[:, :, 2], 0).sum(axis=1)
    cols.append((fwd / np.maximum(n, 1))[:, None])

    out = np.hstack([c if c.ndim == 2 else c[:, None] for c in cols])
    return np.nan_to_num(out).astype(np.float32)


def build_models(seed: int, n_jobs: int):
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "MajorityClass": DummyClassifier(strategy="prior"),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced",
            n_jobs=n_jobs, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=n_jobs, random_state=seed, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.1,
            class_weight="balanced", n_jobs=n_jobs, random_state=seed,
            verbose=-1),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150,
                          early_stopping=True, random_state=seed)),
    }


def evaluate(name, model, Xtr, ytr, Xte, yte, n_classes, sample_weight=None):
    t0 = time.time()
    if sample_weight is not None and name in ("XGBoost",):
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)
    fit_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(Xte)
    infer_ms = (time.time() - t0) / max(len(yte), 1) * 1e3

    prec, rec, f1, sup = precision_recall_fscore_support(
        yte, pred, labels=range(n_classes), zero_division=0)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(yte, pred, average="weighted",
                                      zero_division=0)),
        "per_class": {
            ID_TO_GROUP_LABEL.get(i, str(i)): {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(n_classes)},
        "fit_seconds": round(fit_s, 2),
        "inference_ms_per_flow": round(infer_ms, 4),
        "confusion_matrix": confusion_matrix(
            yte, pred, labels=range(n_classes)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Matched baseline suite")
    ap.add_argument("--data", default="data/sequences/entropy_flows_v2.h5")
    ap.add_argument("--split", default=["flow-isolated"], nargs="+",
                    choices=list(P.SPLIT_MODES),
                    help="Split rules to sweep")
    ap.add_argument("--context", default=["none", "real"], nargs="+",
                    choices=list(P.ENTROPY_MODES),
                    help="Context presentations to sweep")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--subsample-per-class", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default="results/controlled/baselines.json")
    args = ap.parse_args()

    X_seq, X_ent, y, flow_ids = P.load_corpus(args.data, seq_len=args.seq_len)
    if args.subsample_per_class:
        keep = P.stratified_subsample(y, args.subsample_per_class, 0)
        X_seq, X_ent, y, flow_ids = (X_seq[keep], X_ent[keep],
                                     y[keep], flow_ids[keep])
    logger.info("Corpus: %d flows, %d classes", len(y), args.num_classes)

    summary = flow_summary(X_seq)
    logger.info("Flow summary features: %d", summary.shape[1])

    # Individual fits are cached so a long sweep survives an interruption
    # and resumes at the cell it stopped on.
    cache_path = Path(args.out).with_suffix(".partial.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache:
        logger.info("Resuming with %d cached fits", len(cache))

    for split in args.split:
        for seed in args.seeds:
            tr, va, te = P.split_indices(y, flow_ids, X_ent, split, seed)
            # Validation flows are not used by these estimators, so they are
            # folded into training; the test partition is untouched.
            fit_idx = np.concatenate([tr, va])
            logger.info("%s seed %d: fit=%d test=%d",
                        split, seed, len(fit_idx), len(te))

            for context in args.context:
                pending = [(n, m) for n, m in build_models(
                    seed, args.n_jobs).items()
                    if f"{n}|{context}|{split}|{seed}" not in cache]
                if not pending:
                    logger.info("  all estimators cached for context=%s "
                                "split=%s seed=%d", context, split, seed)
                    continue

                ent = P.apply_entropy_mode(
                    X_ent, y, context, seed,
                    reference=X_ent[tr], reference_y=y[tr])
                X = summary if context == "none" else np.hstack([summary, ent])

                norm_mean = X[fit_idx].mean(axis=0)
                norm_std = X[fit_idx].std(axis=0) + 1e-8
                Xn = (X - norm_mean) / norm_std

                classes, counts = np.unique(y[fit_idx], return_counts=True)
                w = {int(c): len(fit_idx) / (len(classes) * n)
                     for c, n in zip(classes, counts)}
                sw = np.array([w[int(c)] for c in y[fit_idx]])

                for name, model in pending:
                    res = evaluate(name, model, Xn[fit_idx], y[fit_idx],
                                   Xn[te], y[te], args.num_classes, sw)
                    cache[f"{name}|{context}|{split}|{seed}"] = res
                    cache_path.write_text(json.dumps(cache))
                    logger.info("  %-20s context=%-15s split=%-16s "
                                "acc=%.4f macro_f1=%.4f",
                                name, context, split,
                                res["accuracy"], res["macro_f1"])

    runs = {}
    for key, res in cache.items():
        name, context, split, _ = key.split("|")
        runs.setdefault(f"{name}|{context}|{split}", []).append(res)

    aggregate = {}
    for key, reps in runs.items():
        name, context, split = key.split("|")
        label = context if split == "flow-isolated" else f"{context}@{split}"
        aggregate.setdefault(name, {})[label] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in reps])),
            "accuracy_std": float(np.std([r["accuracy"] for r in reps])),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in reps])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in reps])),
            "inference_ms_per_flow": float(
                np.median([r["inference_ms_per_flow"] for r in reps])),
            "seeds": args.seeds,
            "runs": reps,
        }

    out = {
        "protocol": {
            "splits": args.split,
            "contexts": args.context,
            "seq_len": args.seq_len,
            "subsample_per_class": args.subsample_per_class,
            "corpus": args.data,
            "seeds": args.seeds,
            "note": "identical corpus, subsample, partition and feature "
                    "construction for every estimator",
        },
        "n_summary_features": int(summary.shape[1]),
        "models": aggregate,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
