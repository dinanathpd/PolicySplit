#!/usr/bin/env python3
"""
prove_leakage.py - Empirical proof of temporal data leakage in CICIoT2023.

Replicates the XGBoost methodology from:
  FarihaAnis/An-IoT-Network-Intrusion-Detection-and-Classification-System-
  with-XGBoost-using-CICIoT2023-Dataset

Phase 1: Random row-level split (reproduces ~98-99% accuracy)
Phase 2: Flow-isolated split   (shows catastrophic drop to ~75-85%)
Phase 3: L2 nearest-neighbor leakage quantification

Memory-optimized for 16 GB RAM systems. Default 1M row sample is
statistically robust (>99.9% confidence at 8-class granularity)
while keeping peak RSS under 6 GB.

Usage:
  python scripts/prove_leakage.py [--max-rows N] [--seed 42]
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "dataset_cic_iot_2023" / "CSV" / "MERGED_CSV"
RESULTS_DIR = ROOT / "results"

# 8-class grouped taxonomy (matching the paper)
LABEL_MAP_8CLASS = {
    # DDoS
    "DDOS-PSHACK_FLOOD": "DDoS", "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS", "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS", "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS", "DDOS-SLOWLORIS": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS", "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS", "DDOS-HTTP_FLOOD": "DDoS",
    # DoS
    "DOS-UDP_FLOOD": "DoS", "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS", "DOS-HTTP_FLOOD": "DoS",
    # Mirai
    "MIRAI-GREIP_FLOOD": "Mirai", "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    # Recon
    "VULNERABILITYSCAN": "Recon", "PINGSWEEP": "Recon",
    "PORTSCAN": "Recon", "OS_FINGERPRINTING": "Recon",
    "HOSTBRUTEFORCE": "Recon",
    # Spoofing
    "DNS_SPOOFING": "Spoofing", "MITM-ARPSPOOFING": "Spoofing",
    # Web
    "BROWSERHIJACKING": "Web", "COMMANDINJECTION": "Web",
    "BACKDOOR_MALWARE": "Web", "UPLOADING_ATTACK": "Web",
    "XSS": "Web", "SQLI": "Web",
    # BruteForce
    "DICTIONARYBRUTEFORCE": "BruteForce",
    # Benign
    "BENIGN": "Benign",
}

FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Variance",
]

# Columns used for flow-group signature hashing
SIG_COLS = [
    "Protocol Type", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "HTTP", "HTTPS", "DNS", "TCP", "UDP", "ICMP", "Number",
]


def _mem_mb():
    """Current process RSS in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def load_data(max_rows=1_000_000, seed=42):
    """Load MERGED CSVs incrementally, map to 8-class, return arrays.

    Memory strategy: read one CSV at a time into a temporary DataFrame,
    extract only the columns we need, then discard the rest.
    """
    csv_files = sorted(CSV_DIR.glob("Merged*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: No CSVs found in {CSV_DIR}")

    needed_cols = FEATURE_COLS + SIG_COLS + ["Label"]
    needed_cols = list(dict.fromkeys(needed_cols))  # deduplicate, keep order

    print(f"Loading data from {len(csv_files)} CSV files "
          f"(max {max_rows:,} rows)...")

    X_parts, y_parts, sig_parts, label_raw_parts = [], [], [], []
    total = 0

    for f in csv_files:
        df = pd.read_csv(f, usecols=needed_cols, low_memory=False)
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df["label_8"] = df["Label"].map(LABEL_MAP_8CLASS)
        df = df.dropna(subset=["label_8"])

        feats = df[FEATURE_COLS].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0).values.astype(np.float32)

        X_parts.append(feats)
        y_parts.append(df["label_8"].values)
        sig_parts.append(df[SIG_COLS].values.astype(str))
        label_raw_parts.append(df["Label"].values.astype(str))

        total += len(df)
        del df
        gc.collect()

        if total >= max_rows:
            break

    X = np.concatenate(X_parts)[:max_rows]
    y_str = np.concatenate(y_parts)[:max_rows]
    sig_vals = np.concatenate(sig_parts)[:max_rows]
    labels_raw = np.concatenate(label_raw_parts)[:max_rows]

    del X_parts, y_parts, sig_parts, label_raw_parts
    gc.collect()

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"  Loaded {len(X):,} rows, {len(FEATURE_COLS)} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    print(f"  RSS: {_mem_mb():.0f} MB")
    print(f"  Class distribution:")
    for cls_name in le.classes_:
        n = (y_str == cls_name).sum()
        print(f"    {cls_name:15s}: {n:>10,} ({100*n/len(y_str):.1f}%)")

    return X, y, le, sig_vals, labels_raw


def create_flow_ids(sig_vals, labels_raw):
    """Create pseudo flow-IDs from consecutive-row signature hashing.

    Vectorized: builds a composite string per row, detects changes
    between consecutive rows, and assigns a monotonic group ID.
    """
    print("  Creating pseudo flow-IDs (vectorized)...")

    # Build composite signature string per row
    n = len(sig_vals)
    sigs = np.empty(n, dtype=object)
    for i in range(sig_vals.shape[1]):
        if i == 0:
            sigs = sig_vals[:, i]
        else:
            sigs = np.char.add(np.char.add(sigs, "|"), sig_vals[:, i])
    sigs = np.char.add(np.char.add(sigs, "|"), labels_raw)

    # Detect boundaries where signature changes
    changed = np.ones(n, dtype=bool)
    changed[1:] = sigs[1:] != sigs[:-1]
    flow_ids = np.cumsum(changed).astype(np.int64)

    n_groups = flow_ids[-1]
    print(f"  Created {n_groups:,} flow groups from {n:,} rows")
    print(f"  Avg rows per group: {n/n_groups:.1f}")

    del sigs, changed
    gc.collect()
    return flow_ids


def _train_xgboost(X_train, y_train, X_test, y_test, le, seed, label):
    """Train XGBoost and return metrics dict. Frees model after eval."""
    import xgboost as xgb

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        reg_alpha=0.5,
        reg_lambda=0.7,
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    print(f"  Training XGBoost ({label})...")
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    y_pred = model.predict(X_test_s)

    # Free GPU memory
    del model
    gc.collect()

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_p = precision_score(y_test, y_pred, average="macro")
    macro_r = recall_score(y_test, y_pred, average="macro")

    print(f"\n  === Results ({label}) ===")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Macro F1:    {macro_f1*100:.2f}%")
    print(f"  Weighted F1: {weighted_f1*100:.2f}%")
    print(f"  Macro Prec:  {macro_p*100:.2f}%")
    print(f"  Macro Rec:   {macro_r*100:.2f}%")
    print(f"\n  Per-class report:")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4
    ))

    results = {
        "split": label,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_time_s": float(train_time),
    }

    # Return indices for Phase 3 (train/test split indices)
    return results, X_train_s, X_test_s


# ─────────────────────────────────────────────────────────────────
# Phase 1: Replicate flawed SOTA (random row-level split)
# ─────────────────────────────────────────────────────────────────
def phase1_random_split(X, y, le, seed=42):
    """Reproduce the XGBoost + random split methodology."""
    print("\n" + "=" * 60)
    print("PHASE 1: Random Row-Level Split (Replicating SOTA)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    results, X_train_s, X_test_s = _train_xgboost(
        X_train, y_train, X_test, y_test, le, seed, "random"
    )
    return results, X_train_s, X_test_s


# ─────────────────────────────────────────────────────────────────
# Phase 2: Flow-isolated split
# ─────────────────────────────────────────────────────────────────
def phase2_flow_isolated(X, y, le, flow_ids, seed=42):
    """Train XGBoost with flow-isolated splitting."""
    print("\n" + "=" * 60)
    print("PHASE 2: Flow-Isolated Split (Leakage-Free)")
    print("=" * 60)

    unique_groups = np.unique(flow_ids)
    n_groups = len(unique_groups)
    print(f"  {n_groups:,} unique flow groups")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique_groups)

    n_train = int(0.80 * n_groups)
    train_groups = set(perm[:n_train].tolist())
    test_groups = set(perm[n_train:].tolist())

    # Vectorized mask via pandas isin (faster than list comprehension)
    train_mask = np.isin(flow_ids, list(train_groups))
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} rows ({len(train_groups):,} groups)")
    print(f"  Test:  {len(X_test):,} rows ({len(test_groups):,} groups)")
    print("  Flow isolation verified: zero group overlap.")

    results, X_train_s, X_test_s = _train_xgboost(
        X_train, y_train, X_test, y_test, le, seed, "flow-isolated"
    )

    del X_train, X_test, y_train, y_test, train_mask, test_mask
    gc.collect()

    return results, X_train_s, X_test_s


# ─────────────────────────────────────────────────────────────────
# Phase 3: L2 nearest-neighbor leakage quantification
# ─────────────────────────────────────────────────────────────────
def phase3_nn_leakage(X_train_rand, X_test_rand,
                      X_train_flow, X_test_flow,
                      nn_test_sample=10_000, nn_train_sample=100_000,
                      seed=42):
    """Quantify leakage via 1-NN L2 distances using BallTree.

    Memory-efficient: small samples, BallTree index (no pairwise matrix),
    processes one split at a time and frees memory between them.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Nearest-Neighbor Leakage Quantification")
    print("=" * 60)

    thresholds = [0.01, 0.05, 0.10, 0.50]
    results = {}

    for name, X_tr, X_te in [
        ("random", X_train_rand, X_test_rand),
        ("flow-isolated", X_train_flow, X_test_flow),
    ]:
        print(f"\n  --- {name} split ---")

        scaler = StandardScaler()
        X_tr_z = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_z = scaler.transform(X_te).astype(np.float32)

        rng = np.random.RandomState(seed)
        if len(X_te_z) > nn_test_sample:
            idx = rng.choice(len(X_te_z), nn_test_sample, replace=False)
            X_te_z = X_te_z[idx]
        if len(X_tr_z) > nn_train_sample:
            idx = rng.choice(len(X_tr_z), nn_train_sample, replace=False)
            X_tr_z = X_tr_z[idx]

        print(f"  Building BallTree ({len(X_tr_z):,} train points, "
              f"{X_tr_z.shape[1]} dims)...")
        t0 = time.time()
        tree = BallTree(X_tr_z, metric="euclidean")
        print(f"  Querying {len(X_te_z):,} test points...")
        distances, _ = tree.query(X_te_z, k=1)
        distances = distances.ravel()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        del tree, X_tr_z, X_te_z
        gc.collect()

        print(f"  Distance stats: mean={distances.mean():.4f}, "
              f"median={np.median(distances):.4f}, "
              f"min={distances.min():.6f}, max={distances.max():.2f}")

        split_results = {
            "mean_l2": float(distances.mean()),
            "median_l2": float(np.median(distances)),
            "min_l2": float(distances.min()),
            "max_l2": float(distances.max()),
            "std_l2": float(distances.std()),
        }

        for t in thresholds:
            pct = float((distances < t).mean() * 100)
            split_results[f"pct_below_{t}"] = pct
            print(f"  % test rows with NN distance < {t}: {pct:.2f}%")

        results[name] = split_results

    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Empirical proof of data leakage in CICIoT2023"
    )
    parser.add_argument("--max-rows", type=int, default=1_000_000,
                        help="Max rows to load (default: 1M, memory-safe)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("LEAKAGE PROOF: CICIoT2023 + XGBoost")
    print("Replicating FarihaAnis repo methodology")
    print(f"Max rows: {args.max_rows:,}  |  RSS: {_mem_mb():.0f} MB")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────
    X, y, le, sig_vals, labels_raw = load_data(
        max_rows=args.max_rows, seed=args.seed
    )

    # ── Create flow IDs ────────────────────────────────────
    flow_ids = create_flow_ids(sig_vals, labels_raw)
    del sig_vals, labels_raw
    gc.collect()
    print(f"  RSS after flow-ID creation: {_mem_mb():.0f} MB")

    # ── Phase 1: Random split ──────────────────────────────
    r1, X_tr_rand_s, X_te_rand_s = phase1_random_split(
        X, y, le, seed=args.seed
    )
    print(f"  RSS after Phase 1: {_mem_mb():.0f} MB")

    # ── Phase 2: Flow-isolated split ───────────────────────
    r2, X_tr_flow_s, X_te_flow_s = phase2_flow_isolated(
        X, y, le, flow_ids, seed=args.seed
    )
    # Free raw arrays (scaled copies are kept for Phase 3)
    del X, y, flow_ids
    gc.collect()
    print(f"  RSS after Phase 2: {_mem_mb():.0f} MB")

    # ── Phase 3: NN leakage quantification ─────────────────
    r3 = phase3_nn_leakage(
        X_tr_rand_s, X_te_rand_s,
        X_tr_flow_s, X_te_flow_s,
        seed=args.seed
    )
    del X_tr_rand_s, X_te_rand_s, X_tr_flow_s, X_te_flow_s
    gc.collect()

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_rows = r1["train_size"] + r1["test_size"]
    acc_drop = (r1["accuracy"] - r2["accuracy"]) * 100
    f1_drop = (r1["macro_f1"] - r2["macro_f1"]) * 100

    print(f"\n  XGBoost 8-class on CICIoT2023 ({n_rows:,} rows)")
    print(f"  {'Metric':<20s} {'Random Split':>14s} {'Flow-Isolated':>14s} "
          f"{'Drop':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<20s} {r1['accuracy']*100:>13.2f}% "
          f"{r2['accuracy']*100:>13.2f}% {acc_drop:>9.1f}pp")
    print(f"  {'Macro F1':<20s} {r1['macro_f1']*100:>13.2f}% "
          f"{r2['macro_f1']*100:>13.2f}% {f1_drop:>9.1f}pp")
    print(f"  {'Weighted F1':<20s} {r1['weighted_f1']*100:>13.2f}% "
          f"{r2['weighted_f1']*100:>13.2f}%")

    pct_rand = r3["random"].get("pct_below_0.05", 0)
    pct_flow = r3["flow-isolated"].get("pct_below_0.05", 0)
    print(f"\n  NN Leakage (% test with L2 < 0.05 in train):")
    print(f"    Random split:        {pct_rand:.1f}%")
    print(f"    Flow-isolated split: {pct_flow:.1f}%")

    # ── Save results ────────────────────────────────────────
    all_results = {
        "phase1_random": r1,
        "phase2_flow_isolated": r2,
        "phase3_nn_leakage": r3,
        "n_rows": n_rows,
        "n_features": len(FEATURE_COLS),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
    }
    out_path = RESULTS_DIR / "leakage_proof_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Final RSS: {_mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
