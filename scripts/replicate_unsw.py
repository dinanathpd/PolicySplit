"""
replicate_unsw.py - Repeat the context decomposition on UNSW-NB15.

UNSW-NB15 ships seven connection-count features computed over a sliding
window of the previous one hundred connections. That is the same construct
as the context channel of the main study: a feature whose support extends
beyond the record it labels, computed over a corpus assembled from
per-category simulation runs. If the decomposition reported for CICIoT2023
is a property of how such features are built rather than of that one
corpus, the same controls should behave the same way here.

Everything except the corpus is held fixed: the same four conditions, the
same substitution routine, statistics estimated after partitioning, and
ten seeds per cell.

    python scripts/replicate_unsw.py
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.protocol import apply_entropy_mode  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# The seven counts UNSW-NB15 computes over the previous 100 connections.
# ct_state_ttl is a state and time-to-live bucket, and ct_ftp_cmd and
# ct_flw_http_mthd are per-flow protocol counters, so none of the three is
# window scoped and all three stay in the base set.
CONTEXT = ["ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm",
           "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"]
CATEGORICAL = ["proto", "service", "state"]
DROP = ["id", "label", "attack_cat"]
CONDITIONS = ("none", "shuffle", "class-marginal", "real")


def load_corpus(root: Path):
    frames = [pd.read_csv(p) for p in sorted(root.glob("UNSW_NB15_*-set.csv"))]
    df = pd.concat(frames, ignore_index=True)
    classes = sorted(df["attack_cat"].unique())
    y = df["attack_cat"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctx = df[CONTEXT].to_numpy(dtype=np.float32)
    base = df.drop(columns=DROP + CONTEXT)
    logger.info("pooled %d rows, %d classes, %d base columns, %d context",
                len(df), len(classes), base.shape[1], len(CONTEXT))
    return base, ctx, y, classes


def build_estimators(seed, n_jobs):
    est = {
        "LogisticRegression": LogisticRegression(
            max_iter=300, n_jobs=n_jobs, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=n_jobs, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        est["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, tree_method="hist",
            n_jobs=n_jobs, random_state=seed, verbosity=0)
    except ImportError:
        logger.warning("xgboost unavailable, skipping")
    try:
        from lightgbm import LGBMClassifier
        est["LightGBM"] = LGBMClassifier(
            n_estimators=200, num_leaves=31, n_jobs=n_jobs,
            random_state=seed, verbose=-1)
    except ImportError:
        logger.warning("lightgbm unavailable, skipping")
    return est


def _policy_corpus(args):
    """Classes (Normal first), labels and the five switch counters."""
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path(args.data).glob("UNSW_NB15_*-set.csv"))],
                   ignore_index=True)
    # Normal first, so index 0 is the allow class as in the main corpus.
    classes = ["Normal"] + sorted(c for c in df["attack_cat"].unique()
                                  if c != "Normal")
    y_all = df["attack_cat"].map(
        {c: i for i, c in enumerate(classes)}).to_numpy()
    raw_all = np.stack([df.sbytes + df.dbytes, df.spkts + df.dpkts, df.smean,
                        df.dmean, np.rint(df.sinpkt * 1000)],
                       axis=1).astype(np.float64)
    return classes, y_all, raw_all


def _policy_seed(T, P, y_all, raw_all, n_classes, seed):
    """Subsample, random partition, fitted table and calibration units."""
    keep = P.stratified_subsample(y_all, 12000, seed)
    y, raw = y_all[keep], raw_all[keep]
    tr, va, te = P.split_indices(y, np.arange(len(y)).astype(str),
                                 np.zeros((len(y), 1)), "random", seed)
    order = np.argsort(va, kind="stable")    # file row order
    # A random partition assigns single rows, so rows are the units.
    clusters = T.cluster_weights(np.arange(len(va)))
    F = T.apply_edges(raw, T.fit_edges(raw[tr]))
    c_tr, c_va, c_te = (T.cell_index(F[i]) for i in (tr, va, te))
    weights = {c: len(tr) / (n_classes * n) for c, n in
               zip(*np.unique(y[tr], return_counts=True))}
    leaf, leaf_conf, _ = T.empirical_table(F[tr], y[tr], n_classes, 0,
                                           weights)
    counts = np.zeros((T.KEY_SPACE, n_classes))
    np.add.at(counts, (c_tr, y[tr]), 1.0)
    return (y, va, te, order, clusters, c_va, c_te, leaf, leaf_conf, counts)


def map_replication(args):
    """The random action-map study of train_rf.py --map-study on UNSW-NB15."""
    sys.path.insert(0, os.path.dirname(__file__))
    import train_rf as T
    from src.preprocessing import protocol as P
    T.set_key_levels(T.KEY_RADIX)
    classes, y_all, raw_all = _policy_corpus(args)
    n = len(classes)
    act4 = {"Normal": 0, "Backdoor": 1, "DoS": 1, "Exploits": 1,
            "Shellcode": 1, "Worms": 1, "Fuzzers": 2, "Generic": 2,
            "Analysis": 3, "Reconnaissance": 3}
    maps = ([np.array([0] + [1] * (n - 1)), np.array([act4[c] for c in classes])]
            + T.random_action_maps(n, 2, args.map_count, 1)
            + T.random_action_maps(n, 3, args.map_count, 2) + [np.arange(n)])
    uniq, seen = [], set()
    for m in maps:
        names = {}
        key = tuple(names.setdefault(v, len(names)) for v in m.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    seeds = range(args.policy_first_seed,
                  args.policy_first_seed + args.policy_seeds)
    per_seed = []
    for seed in seeds:
        (y, va, te, order, clusters, c_va, c_te, leaf, leaf_conf,
         counts) = _policy_seed(T, P, y_all, raw_all, n, seed)
        per_seed.append({"seed": seed, "maps": T.map_gains(
            counts, leaf, leaf_conf, c_va, c_te, y[va], y[te], uniq,
            args.map_target, 0.05, order, clusters)})
        logger.info("UNSW map study seed %d done (%d maps)", seed, len(uniq))
    out = {"config": {"corpus": "UNSW-NB15, stratified random partitions",
                      "classes": classes, "target": args.map_target,
                      "seeds": list(seeds), "key_levels": T.KEY_RADIX,
                      "random_maps_per_size": args.map_count,
                      "calibration": "dependence rule, rows as units"},
           "summary": T.summarise_maps(per_seed, args.map_target),
           "per_seed": per_seed}
    Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.map_out).write_text(json.dumps(out))
    logger.info("written to %s: rho %.2f (aggregation), %.2f (crossing)",
                args.map_out, out["summary"]["spearman_rho"],
                out["summary"]["crossing_rho"])


def policy_replication(args):
    """
    Label against action admission on counters a switch can keep per flow.
    UNSW-NB15 has no flow identifier, so partitions are stratified random.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    import train_rf as T
    from src.preprocessing import protocol as P

    classes, y_all, raw_all = _policy_corpus(args)
    # Two-action policy, and a four-action one in the spirit of the main
    # corpus: drop intrusions, rate-limit noisy probing and fuzzing, and
    # redirect reconnaissance and analysis traffic for inspection.
    act4 = {"Normal": 0, "Backdoor": 1, "DoS": 1, "Exploits": 1,
            "Shellcode": 1, "Worms": 1, "Fuzzers": 2, "Generic": 2,
            "Analysis": 3, "Reconnaissance": 3}
    maps = {"2-action": np.array([0] + [1] * (len(classes) - 1)),
            "4-action": np.array([act4[c] for c in classes])}
    budgets = (0.005, 0.01, 0.02, 0.05, 0.10)
    rows = []
    seeds = range(args.policy_first_seed,
                  args.policy_first_seed + args.policy_seeds)
    for seed in seeds:
        keep = P.stratified_subsample(y_all, 12000, seed)
        y, raw = y_all[keep], raw_all[keep]
        tr, va, te = P.split_indices(y, np.arange(len(y)).astype(str),
                                     np.zeros((len(y), 1)), "random", seed)
        order = np.argsort(va, kind="stable")    # file row order
        # A random partition assigns single rows, so rows are the units.
        clusters = T.cluster_weights(np.arange(len(va)))
        F = T.apply_edges(raw, T.fit_edges(raw[tr]))
        c_tr, c_va, c_te = (T.cell_index(F[i]) for i in (tr, va, te))
        weights = {c: len(tr) / (len(classes) * n) for c, n in
                   zip(*np.unique(y[tr], return_counts=True))}
        leaf, leaf_conf, _ = T.empirical_table(F[tr], y[tr], len(classes), 0,
                                               weights)
        counts = np.zeros((T.KEY_SPACE, len(classes)))
        np.add.at(counts, (c_tr, y[tr]), 1.0)
        row = {"split": "random", "seed": seed, "gates": {}, "tiered": {},
               "tiered_blk": {}, "tiered_dep": {},
               "aggregation_mass": {m: T.aggregation_mass(counts, a)
                                    for m, a in maps.items()},
               "confusion": {m: T.confusion_structure(counts, a, classes)
                             for m, a in maps.items()}}
        for mname, amap in maps.items():
            row["gates"][mname] = {}
            for g, (score, act) in T.cell_gates(counts, leaf, leaf_conf,
                                                amap).items():
                res, _ = T.gate_outcomes(score[c_va], act[c_va], amap[y[va]],
                                         score[c_te], act[c_te], amap[y[te]],
                                         budgets, 0.05, order=order,
                                         groups=clusters)
                row["gates"][mname][g] = {str(k): v for k, v in res.items()}
                if mname != "2-action" and g in ("label", "action"):
                    for tk, o, c, rule in (
                            ("tiered", None, None, "dependence"),
                            ("tiered_blk", order, None, "block"),
                            ("tiered_dep", order, clusters, "dependence")):
                        row[tk].setdefault(mname, {})[g] = T.tiered_outcomes(
                            score[c_va], act[c_va], amap[y[va]], score[c_te],
                            act[c_te], amap[y[te]], 0.05, o, c, rule)
        rows.append(row)
        g2 = row["gates"]["2-action"]
        logger.info("UNSW policy seed %d: 1%% coverage label %.1f%% action %.1f%%"
                    " (dependence rule %.1f%% vs %.1f%%)", seed,
                    100 * g2["label"]["0.01"]["cal_cov"],
                    100 * g2["action"]["0.01"]["cal_cov"],
                    100 * g2["label"]["0.01"]["dep_cov"],
                    100 * g2["action"]["0.01"]["dep_cov"])
    out = {"corpus": "UNSW-NB15, pooled partitions", "classes": classes,
           "features": ["sbytes+dbytes", "spkts+dpkts", "smean", "dmean",
                        "sinpkt_us"],
           "key_levels": T.KEY_RADIX, "seeds": list(seeds),
           "action_maps": {k: v.tolist() for k, v in maps.items()},
           "note": "end-of-flow counters, stratified random partitions; block "
                   "calibration uses file row order; the dependence rule "
                   "resamples rows, the units a random partition assigns",
           "summary": T.summarise_study(rows, budgets),
           "per_seed": rows}
    Path(args.policy_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.policy_out).write_text(json.dumps(out))
    logger.info("written to %s", args.policy_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset_unsw_nb15")
    ap.add_argument("--out", default="results/unsw_nb15/decomposition.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--policy", action="store_true",
                    help="Replicate the admission-rule study instead")
    ap.add_argument("--policy-seeds", type=int, default=30)
    ap.add_argument("--policy-first-seed", type=int, default=30)
    ap.add_argument("--policy-out", default="results/unsw_nb15/policy.json")
    ap.add_argument("--map-study", action="store_true",
                    help="Gain of the action gate over random action maps")
    ap.add_argument("--map-out", default="results/unsw_nb15/map_study.json")
    ap.add_argument("--map-count", type=int, default=30)
    ap.add_argument("--map-target", type=float, default=0.01)
    args = ap.parse_args()

    if args.policy:
        policy_replication(args)
        return
    if args.map_study:
        map_replication(args)
        return

    # sklearn and the boosters both parallelise; letting BLAS do so as well
    # oversubscribes the cores and costs an order of magnitude.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    base, ctx, y, classes = load_corpus(Path(args.data))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.json")
    done = json.loads(partial.read_text()) if partial.exists() else {}

    numeric = [c for c in base.columns if c not in CATEGORICAL]
    for seed in args.seeds:
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=args.test_size, random_state=seed,
            stratify=y)

        # every transform is fitted on the training partition alone
        pre = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  max_categories=32, sparse_output=False),
             CATEGORICAL)])
        Xb_tr = pre.fit_transform(base.iloc[idx_tr]).astype(np.float32)
        Xb_te = pre.transform(base.iloc[idx_te]).astype(np.float32)

        cs = StandardScaler().fit(ctx[idx_tr])
        for condition in CONDITIONS:
            # substitutes are drawn from the training partition only
            c_tr = apply_entropy_mode(ctx[idx_tr], y[idx_tr], condition, seed,
                                      reference=ctx[idx_tr],
                                      reference_y=y[idx_tr])
            c_te = apply_entropy_mode(ctx[idx_te], y[idx_te], condition, seed,
                                      reference=ctx[idx_tr],
                                      reference_y=y[idx_tr])
            X_tr = np.hstack([Xb_tr, cs.transform(c_tr).astype(np.float32)])
            X_te = np.hstack([Xb_te, cs.transform(c_te).astype(np.float32)])

            for name, model in build_estimators(seed, args.n_jobs).items():
                key = f"{name}|{condition}|{seed}"
                if key in done:
                    continue
                t0 = time.time()
                model.fit(X_tr, y[idx_tr])
                pred = model.predict(X_te)
                done[key] = {
                    "accuracy": float(accuracy_score(y[idx_te], pred)),
                    "macro_f1": float(f1_score(y[idx_te], pred,
                                               average="macro")),
                    "fit_seconds": round(time.time() - t0, 1),
                }
                logger.info("%-18s %-15s s%d  acc=%.4f f1=%.4f  (%.0fs)",
                            name, condition, seed, done[key]["accuracy"],
                            done[key]["macro_f1"], done[key]["fit_seconds"])
                partial.write_text(json.dumps(done, indent=1))

    models = sorted({k.split("|")[0] for k in done})
    summary = {}
    for name in models:
        entry = {}
        for condition in CONDITIONS:
            vals = [v["macro_f1"] for k, v in done.items()
                    if k.startswith(f"{name}|{condition}|")]
            acc = [v["accuracy"] for k, v in done.items()
                   if k.startswith(f"{name}|{condition}|")]
            if vals:
                entry[condition] = {
                    "macro_f1_mean": float(np.mean(vals)),
                    "macro_f1_std": float(np.std(vals)),
                    "accuracy_mean": float(np.mean(acc)),
                    "n_seeds": len(vals),
                    "by_seed": {k.split("|")[2]: v["macro_f1"]
                                for k, v in done.items()
                                if k.startswith(f"{name}|{condition}|")},
                }
        if {"none", "real", "class-marginal", "shuffle"} <= set(entry):
            gain = 100 * (entry["real"]["macro_f1_mean"]
                          - entry["none"]["macro_f1_mean"])
            prior = 100 * (entry["class-marginal"]["macro_f1_mean"]
                           - entry["none"]["macro_f1_mean"])
            caps = 100 * (entry["shuffle"]["macro_f1_mean"]
                          - entry["none"]["macro_f1_mean"])
            entry["decomposition"] = {
                "total_context_gain_pp": round(gain, 2),
                "explained_by_extra_inputs_pp": round(caps, 2),
                "explained_by_class_marginal_pp": round(prior, 2),
                "beyond_class_marginal_pp": round(gain - prior, 2),
            }
        summary[name] = entry

    out_path.write_text(json.dumps({
        "corpus": "UNSW-NB15, pooled partitions",
        "n_rows": int(len(y)),
        "classes": classes,
        "context_features": CONTEXT,
        "context_window": "previous 100 connections",
        "protocol": {
            "split": f"stratified random, test_size={args.test_size}",
            "seeds": args.seeds,
            "conditions": list(CONDITIONS),
            "note": "controls drawn from the training partition only; "
                    "all scaling and encoding fitted after partitioning",
        },
        "models": summary,
    }, indent=2))
    logger.info("written to %s", out_path)
    for name, e in summary.items():
        d = e.get("decomposition")
        if d:
            logger.info("%-18s gain %+.1f  prior %+.1f  beyond %+.1f", name,
                        d["total_context_gain_pp"],
                        d["explained_by_class_marginal_pp"],
                        d["beyond_class_marginal_pp"])


if __name__ == "__main__":
    main()
