"""
train_rf.py - Fit and compile the PolicySplit fast-path admission policy.

The switch can only compute a handful of statistics at line rate. The
policy comparison uses five flow-local registers evaluated when the tenth
packet arrives:

    * byte count,
    * exponentially weighted moving averages of packet length and
        inter-arrival time, using the same integer shift recurrence,
    * minimum and maximum packet length.

Each statistic is quantised with boundaries fitted on the training
partition. Fine labels are mapped to allow or drop, evidence is aggregated
by action, and only attack cells admitted by the selected false-drop target
are emitted as exact match-action entries. The same script emits the P4
``bucketise()`` action, so offline and switch quantizers use identical,
integer-representable cut points. Source-diversity and port-fanout sketches
remain in the P4 telemetry and alert path, but they are not policy-table
features because the retained corpus cannot reproduce their causal state.

Usage:
    PYTHONPATH=. python scripts/train_rf.py \
        --data data/sequences/entropy_flows_v2.h5 --split flow-isolated
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import protocol as P              # noqa: E402
from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Must match feature_extraction.p4.
EWMA_SHIFT = 3
SKETCH_WIDTH = 32
DECISION_PACKETS = 10
BYTE_EDGES = (512, 1024, 2048, 4096, 8192)
LEN_EDGES = (64, 128, 256, 512, 1024)
IAT_EDGES = (10, 100, 1000, 10000, 100000)      # microseconds
MINLEN_EDGES = (40, 64, 128, 256, 512)
MAXLEN_EDGES = (64, 128, 256, 512, 1024)
FEATURE_NAMES = ("byte_bucket", "len_bucket", "iat_bucket",
                 "minlen_bucket", "maxlen_bucket")
BENIGN_CLASS = 0
ACTION_ALLOW = 0
ACTION_DROP = 1


def integer_ewma(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Replicate the switch recurrence e <- e - (e >> k) + (v >> k).

    ``values`` is (N, L) of non-negative integers and ``valid`` marks the
    observed positions. The first observation seeds the average, matching
    the ``ewma == 0`` branch in the P4 action.
    """
    n, length = values.shape
    out = np.zeros(n, dtype=np.int64)
    started = np.zeros(n, dtype=bool)
    for t in range(length):
        v = values[:, t]
        m = valid[:, t]
        seed = m & ~started
        out[seed] = v[seed]
        started |= seed
        upd = m & started & ~seed
        out[upd] = (out[upd] - (out[upd] >> EWMA_SHIFT)
                    + (v[upd] >> EWMA_SHIFT))
    return out


def linear_count(distinct: np.ndarray, width: int = SKETCH_WIDTH) -> np.ndarray:
    """
    Expected number of set bits when ``distinct`` items are hashed into a
    ``width``-bit bitmap. This is the quantity the switch counter holds,
    including its saturation, so the offline feature and the switch
    feature agree even when the true cardinality exceeds the sketch.
    """
    d = np.clip(distinct, 0, None)
    return width * (1.0 - np.power(1.0 - 1.0 / width, d))


def bucketise(x: np.ndarray, edges) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), x, side="right").astype(np.int64)


def switch_raw(X_seq: np.ndarray, X_ent: np.ndarray) -> np.ndarray:
    """Ten-packet P4 register values in the table-key column order."""
    del X_ent  # context features are intentionally absent from the policy
    if X_seq.shape[1] < DECISION_PACKETS:
        raise ValueError(f"policy requires {DECISION_PACKETS} packets")
    X_seq = X_seq[:, :DECISION_PACKETS]
    valid = X_seq[:, :, 0] != 0
    pkt_count = valid.sum(axis=1)
    if np.any(pkt_count < DECISION_PACKETS):
        raise ValueError("all policy flows must reach the decision boundary")

    pkt_len = np.rint(np.expm1(X_seq[:, :, 0])).astype(np.int64)
    iat_us = np.rint(np.expm1(X_seq[:, :, 1]) * 1000.0).astype(np.int64)

    byte_count = pkt_len.sum(axis=1)
    len_ewma = integer_ewma(pkt_len, valid)
    # The first packet of a flow has no predecessor, so its inter-arrival
    # sample is not defined; the switch only updates the average from the
    # second packet onwards.
    iat_valid = valid.copy()
    iat_valid[np.arange(len(valid)), valid.argmax(axis=1)] = False
    iat_ewma = integer_ewma(iat_us, iat_valid)
    min_len = np.where(valid, pkt_len, np.iinfo(np.int64).max).min(axis=1)
    max_len = pkt_len.max(axis=1)

    return np.stack([byte_count, len_ewma, iat_ewma, min_len, max_len],
                    axis=1).astype(np.float64)


FIXED_EDGES = (BYTE_EDGES, LEN_EDGES, IAT_EDGES,
               MINLEN_EDGES, MAXLEN_EDGES)


def fit_edges(raw_tr: np.ndarray, n_buckets: int = 6):
    """
    Equal-frequency cut points, read off the training partition alone.

    The hand-picked edges leave the table almost empty: they place 98% of
    flows in one source-diversity bucket and use three of six count
    buckets, so the switch discards resolution the table could have held.
    Quantile edges spread the corpus across the key space. They are
    constants compiled into the P4 program, so this costs the switch
    nothing at run time.
    """
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    edges = []
    for j in range(raw_tr.shape[1]):
        # P4 compares integer register values. Quantize the boundaries before
        # fitting so host and switch cannot disagree across a fractional cut,
        # and collapse duplicate integer thresholds.
        cuts = np.unique(np.ceil(np.quantile(raw_tr[:, j], qs))).astype(int)
        edges.append(tuple(float(c) for c in cuts))
    return tuple(edges)


def apply_edges(raw: np.ndarray, edges) -> np.ndarray:
    return np.stack([bucketise(raw[:, j], edges[j])
                     for j in range(raw.shape[1])], axis=1)


def switch_features(X_seq: np.ndarray, X_ent: np.ndarray,
                    edges=FIXED_EDGES) -> np.ndarray:
    """Quantised feature matrix, in the column order of the P4 table key."""
    return apply_edges(switch_raw(X_seq, X_ent), edges)


def export_entries(cls: np.ndarray, conf: np.ndarray,
                   confidence_floor: float, n_classes: int,
                   out_dir: Path) -> dict:
    """
    Emit one match-action entry per cell whose posterior clears the floor.
    Every key field takes six values, so the key space is enumerable and
    each entry is an exact ternary rule; no interval-to-mask approximation
    is involved.
    """
    grid = np.array(np.meshgrid(*[np.arange(KEY_RADIX)] * 5, indexing="ij"))
    keys = grid.reshape(5, -1).T
    keep = conf >= confidence_floor
    lines = ["# table_add <table> <action> <key> => <args> [priority]",
             "# key fields: " + " ".join(FEATURE_NAMES)]
    for k, c, q in zip(keys[keep], cls[keep], conf[keep]):
        key = " ".join(f"{int(v)}&&&0xFF" for v in k)
        lines.append("table_add SplitMambaIngress.flow_classifier set_verdict "
                     f"{key} => {int(c)} {int(round(q * 255))} 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classifier_entries.txt").write_text("\n".join(lines) + "\n")

    summary = {
        "key_space": int(len(keys)),
        "entries_installed": int(keep.sum()),
        "entries_defaulted_to_escalation": int((~keep).sum()),
        "confidence_floor": confidence_floor,
        "class_distribution": np.bincount(
            cls[keep], minlength=n_classes).tolist(),
    }
    (out_dir / "classifier_entries.json").write_text(
        json.dumps(summary, indent=2))
    return summary


MEASURED_FEATURES = ("pkt_count", "byte_count", "mean_iat_us",
                     "dstport_fanout", "src_diversity")


KEY_RADIX = 6
KEY_SPACE = KEY_RADIX ** 5


def cell_index(F: np.ndarray) -> np.ndarray:
    """Fold the five quantised fields into one index over the key space."""
    idx = np.zeros(len(F), dtype=np.int64)
    for j in range(F.shape[1]):
        idx = idx * KEY_RADIX + np.clip(F[:, j], 0, KEY_RADIX - 1)
    return idx


def wilson_lower(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    """Lower bound of a Wilson interval on k/n, and 0 where n is 0."""
    n = np.maximum(n, 1e-9)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.clip(centre - margin, 0.0, 1.0)


def empirical_table(F_tr, y_tr, n_classes, min_support, weights):
    """
    Class posterior of every cell of the key space, from the training rows.

    The ternary table holds 8192 entries and the key space is 7776 cells, so
    the table can carry the empirical posterior exactly. Compiling a
    depth-limited tree instead merges cells that the hardware could have
    kept apart, which is a needless approximation here.

    The verdict is the class-balanced argmax, so rare classes are not
    swamped, but the confidence that gates escalation is the lower bound of
    a Wilson interval on the realised purity of that verdict. That is the
    quantity the escalation decision needs, namely how often answering this
    class in this cell is correct, and it makes a cell seen twice unable to
    claim certainty.
    """
    w = np.array([weights.get(c, 1.0) for c in range(n_classes)])
    counts = np.zeros((KEY_SPACE, n_classes))
    np.add.at(counts, (cell_index(F_tr), y_tr), 1.0)
    support = counts.sum(axis=1)
    balanced = counts * w[None, :]
    total = balanced.sum(axis=1, keepdims=True)
    posterior = np.divide(balanced, total, out=np.zeros_like(balanced),
                          where=total > 0)
    cls = posterior.argmax(axis=1)
    hits = counts[np.arange(KEY_SPACE), cls]
    conf = wilson_lower(hits, support)
    conf[support < min_support] = 0.0
    return cls, conf, support


def action_confidence(F_tr, y_tr, cell_cls, min_support):
    """Purity after mapping fine-grained labels to allow or drop."""
    counts = np.zeros((KEY_SPACE, int(y_tr.max()) + 1))
    np.add.at(counts, (cell_index(F_tr), y_tr), 1.0)
    support = counts.sum(axis=1)
    allow_hits = counts[:, BENIGN_CLASS]
    drop_hits = counts[:, 1:].sum(axis=1)
    action = np.where(drop_hits >= allow_hits, ACTION_DROP, ACTION_ALLOW)
    hits = np.maximum(allow_hits, drop_hits)
    conf = wilson_lower(hits, support)
    conf[support < min_support] = 0.0
    dominant_attack = counts[:, 1:].argmax(axis=1) + 1
    policy_cls = np.where(action == ACTION_DROP, dominant_attack, BENIGN_CLASS)
    return action, conf, policy_cls


def operating_points(conf_te, cls_te, y_te, controller_pred, floors):
    """Resolved fraction and composed accuracy across escalation thresholds."""
    out = []
    for f in floors:
        resolved = conf_te >= f
        composed = np.where(resolved, cls_te, controller_pred)
        out.append({
            "confidence_floor": round(float(f), 3),
            "resolved_fraction": float(resolved.mean()),
            "accuracy_on_resolved": float(
                accuracy_score(y_te[resolved], cls_te[resolved]))
            if resolved.any() else None,
            "composed_accuracy": float(accuracy_score(y_te, composed)),
            "composed_macro_f1": float(
                f1_score(y_te, composed, average="macro", zero_division=0)),
        })
    return out


def evaluate_measured(path: Path, seed: int, num_classes: int) -> dict:
    """
    Evaluate the features the switch itself computed.

    Every value here was produced inside the pipeline from packets that had
    already arrived, so unlike the offline windowed features it carries no
    dependence on the future and none on which capture file a flow belongs
    to. The same controls used on the offline channel are applied to the
    source-diversity estimate: removing it, and replacing it with a draw
    from its own class's distribution.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rec = json.loads(path.read_text())
    meta, rec = rec, rec["records"]
    if not rec:
        raise SystemExit(f"no records in {path}")

    y = np.array([r["label"] for r in rec], dtype=np.int64)
    X = np.array([[float(r[f]) for f in MEASURED_FEATURES] for r in rec],
                 dtype=np.float64)
    div = MEASURED_FEATURES.index("src_diversity")
    truth = np.array([float(r.get("true_victim_sources", 0)) for r in rec])

    logger.info("Measured records: %d flows over %d classes",
                len(y), len(np.unique(y)))

    # Does the in-switch sketch estimate what it claims to estimate?
    ok = truth > 0
    estimator_r = float(np.corrcoef(X[ok, div], truth[ok])[0, 1]) \
        if ok.sum() > 2 else None
    if estimator_r is not None:
        logger.info("linear-counting estimate vs true distinct sources: "
                    "r=%+.3f over %d flows", estimator_r, int(ok.sum()))

    # Descriptive: does the causal estimate separate the two flood families,
    # and does the quantity it estimates separate them either?
    per_class = {}
    for c in np.unique(y):
        v, t = X[y == c, div], truth[y == c]
        per_class[int(c)] = {
            "n": int(len(v)),
            "switch_estimate_mean": float(v.mean()),
            "switch_estimate_median": float(np.median(v)),
            "true_sources_mean": float(t.mean()),
        }

    def contrast(a, b, n_boot=2000):
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        # Percentile bootstrap over flows, so the interval reflects the
        # sample actually collected rather than an asymptotic assumption.
        rs = np.random.RandomState(0)
        boot = np.empty(n_boot)
        for k in range(n_boot):
            ai = a[rs.randint(0, len(a), len(a))]
            bi = b[rs.randint(0, len(b), len(b))]
            boot[k] = mannwhitneyu(ai, bi, alternative="two-sided")[0] \
                / (len(ai) * len(bi))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "cohens_d": float((a.mean() - b.mean()) / pooled),
                "auc": float(auc), "auc_ci95": [float(lo), float(hi)],
                "p": float(p), "n_a": int(len(a)), "n_b": int(len(b))}

    separation = None
    if (y == 1).sum() > 1 and (y == 2).sum() > 1:
        separation = {
            "switch_estimate": contrast(X[y == 1, div], X[y == 2, div]),
            "ground_truth": contrast(truth[y == 1], truth[y == 2]),
        }
        for k, v in separation.items():
            logger.info("DDoS vs DoS %-16s %.2f vs %.2f  d=%+.2f "
                        "AUC=%.3f [%.3f, %.3f] p=%.3g", k, v["mean_a"],
                        v["mean_b"], v["cohens_d"], v["auc"],
                        *v["auc_ci95"], v["p"])

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = order[:cut], order[cut:]

    def score(Xv):
        m = HistGradientBoostingClassifier(
            max_depth=6, max_iter=200, random_state=seed).fit(Xv[tr], y[tr])
        p = m.predict(Xv[te])
        return {"accuracy": float(accuracy_score(y[te], p)),
                "macro_f1": float(f1_score(y[te], p, average="macro",
                                           zero_division=0))}

    # none: the diversity column is blanked. class-marginal: each flow keeps
    # a diversity value from its own class but not its own.
    X_none = X.copy()
    X_none[:, div] = 0.0
    X_marg = X.copy()
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        X_marg[idx, div] = X[rng.choice(idx, len(idx), replace=True), div]

    conditions = {"measured": score(X), "none": score(X_none),
                  "class-marginal": score(X_marg)}
    for k, v in conditions.items():
        logger.info("  diversity=%-15s acc=%.4f macro_f1=%.4f",
                    k, v["accuracy"], v["macro_f1"])

    gain = 100 * (conditions["measured"]["macro_f1"]
                  - conditions["none"]["macro_f1"])
    beyond = 100 * (conditions["measured"]["macro_f1"]
                    - conditions["class-marginal"]["macro_f1"])
    logger.info("diversity worth %.1f pp, of which %.1f pp survives the "
                "class-marginal control", gain, beyond)

    return {
        "source": str(path),
        "n_flows": int(len(y)),
        "packets_replayed": meta.get("packets_sent"),
        "distinct_sources_per_capture": meta.get("distinct_sources_per_capture"),
        "distinct_victims": meta.get("distinct_victims"),
        "features": list(MEASURED_FEATURES),
        "class_counts": np.bincount(y, minlength=num_classes).tolist(),
        "estimator_vs_truth_pearson_r": estimator_r,
        "src_diversity_by_class": per_class,
        "ddos_dos_separation": separation,
        "conditions": conditions,
        "diversity_gain_pp": round(gain, 2),
        "diversity_gain_beyond_class_marginal_pp": round(beyond, 2),
    }


P4_FIELDS = (("byte_bucket", "byte_count"), ("len_bucket", "mean_len"),
             ("iat_bucket", "mean_iat_us"), ("minlen_bucket", "min_len"),
             ("maxlen_bucket", "max_len"))


def export_bucketise(edges, out_dir: Path) -> None:
    """
    Emit the P4 action that quantises the registers.

    The switch-side and host-side boundaries must be identical or the table
    is keyed on something the host never fitted. ``fit_edges`` has already
    converted every boundary to a distinct integer representable by P4.
    """
    order = {n: i for i, n in enumerate(FEATURE_NAMES)}
    lines = ["// Generated by scripts/train_rf.py. Do not edit.",
             "action bucketise() {"]
    for bucket, register in P4_FIELDS:
        cuts = edges[order[bucket]]
        lines.append(f"    meta.{bucket} = 0;")
        for level, cut in reversed(list(enumerate(cuts, start=1))):
            kw = "if     " if level == len(cuts) else "else if"
            lines.append(f"    {kw} (meta.{register} >= {int(np.ceil(cut))})"
                         f" {{ meta.{bucket} = {level}; }}")
        lines.append("")
    lines.append("}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bucketise.p4inc").write_text("\n".join(lines) + "\n")


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided upper confidence bound on a binomial rate, exact."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def calibrate_threshold(conf_cal, pred_cal, y_cal, epsilon, delta, grid):
    """
    Lowest threshold whose flow-level calibration bound is below ``epsilon``.

    Thresholds are tested in a pre-specified descending order, which walks
    from the most conservative setting towards more coverage, and testing
    stops at the first threshold that fails. The binomial bound is exact for
    independent or exchangeable calibration units. CICIoT2023 flows within
    one capture are dependent, so this project treats it as a conservative
    selection rule and audits the selected threshold on a held-out partition.
    """
    chosen, certified = None, None
    for tau in sorted(grid, reverse=True):
        sel = conf_cal >= tau
        n = int(sel.sum())
        if n == 0:
            continue
        k = int((pred_cal[sel] != y_cal[sel]).sum())
        upper = clopper_pearson_upper(k, n, delta)
        if upper <= epsilon:
            chosen, certified = float(tau), upper
        else:
            break
    return chosen, certified


def drop_policy_points(class_conf_cal, class_pred_cal,
                       action_conf_cal, action_pred_cal, y_cal,
                       class_conf_te, class_pred_te,
                       action_conf_te, action_pred_te, y_te,
                       controller_pred, epsilons, delta, grid):
    """Compare label confidence with policy-aligned confidence fairly."""
    y_action_cal = (y_cal != BENIGN_CLASS).astype(np.int64)
    y_action_te = (y_te != BENIGN_CLASS).astype(np.int64)
    controller_action = (controller_pred != BENIGN_CLASS).astype(np.int64)
    out = []
    for eps in epsilons:
        row = {"false_drop_budget": float(eps), "delta": float(delta)}
        for name, cal_score, cal_action, test_score, test_action in (
            ("label_confidence", class_conf_cal, class_pred_cal,
             class_conf_te, class_pred_te),
            ("action_confidence", action_conf_cal, action_pred_cal,
             action_conf_te, action_pred_te)):
            eligible_cal = cal_action == ACTION_DROP
            score_cal = np.where(eligible_cal, cal_score, -1.0)
            tau, upper = calibrate_threshold(
                score_cal, cal_action, y_action_cal, eps, delta, grid)
            if tau is None:
                row[name] = {"threshold": None, "coverage": 0.0,
                             "false_drop_rate": None,
                             "calibration_upper_bound": None}
                continue
            selected = ((test_action == ACTION_DROP) &
                        (test_score >= tau))
            false_drop = (float((y_action_te[selected] == ACTION_ALLOW).mean())
                          if selected.any() else None)
            composed_action = np.where(
                selected, ACTION_DROP, controller_action)
            row[name] = {
                "threshold": float(tau),
                "coverage": float(selected.mean()),
                "false_drop_rate": false_drop,
                "calibration_upper_bound": float(upper),
                "action_accuracy": float(
                    accuracy_score(y_action_te, composed_action)),
            }
        out.append(row)
    return out


def build_controller(name, seed, weights):
    """Adjudicator for escalated flows, chosen by the matched comparison."""
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=6,
                              class_weight=weights, random_state=seed,
                              verbose=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=6,
                             tree_method="hist", n_jobs=6,
                             random_state=seed, verbosity=0)
    return RandomForestClassifier(n_estimators=200, n_jobs=6,
                                  class_weight=weights, random_state=seed)


def main():
    ap = argparse.ArgumentParser(description="Data-plane classifier")
    ap.add_argument("--data", default="data/sequences/entropy_flows_v2.h5")
    ap.add_argument("--split", default="flow-isolated",
                    choices=list(P.SPLIT_MODES))
    ap.add_argument("--seq-len", type=int, default=DECISION_PACKETS)
    ap.add_argument("--subsample-per-class", type=int, default=12000)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-samples-leaf", type=int, default=50)
    ap.add_argument("--table-mode", default="empirical",
                    choices=("empirical", "tree"),
                    help="empirical carries the per-cell posterior, which the "
                         "table is large enough to hold exactly")
    ap.add_argument("--min-cell-support", type=int, default=0)
    ap.add_argument("--edges", default="quantile",
                    choices=("quantile", "fixed"),
                    help="quantile fits equal-frequency cut points on the "
                         "training partition; fixed uses the hand-picked ones")
    ap.add_argument("--controller", default="LightGBM",
                    choices=("LightGBM", "XGBoost", "RandomForest"))
    ap.add_argument("--confidence-floor", type=float, default=0.70)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="tail probability used by the conservative bound")
    ap.add_argument("--false-drop-budgets", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05, 0.10],
                    help="selective drop-error targets used for comparison")
    ap.add_argument("--deploy-false-drop", type=float, default=0.01,
                    help="false-drop budget compiled into the P4 entries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--out-dir", default="p4/generated")
    ap.add_argument("--results", default="results/dataplane_classifier.json")
    ap.add_argument("--from-bmv2", default=None,
                    help="Evaluate the features the switch measured, from a "
                         "collection produced by replay_attack.py --collect")
    args = ap.parse_args()

    if args.from_bmv2:
        out = evaluate_measured(Path(args.from_bmv2), args.seed,
                                args.num_classes)
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(json.dumps(out, indent=2))
        logger.info("Wrote %s", args.results)
        return

    X_seq, X_ent, y, flow_ids = P.load_corpus(args.data, seq_len=args.seq_len)
    if args.subsample_per_class:
        keep = P.stratified_subsample(y, args.subsample_per_class, args.seed)
        X_seq, X_ent, y, flow_ids = (X_seq[keep], X_ent[keep],
                                     y[keep], flow_ids[keep])
    logger.info("Corpus: %d flows", len(y))

    raw = switch_raw(X_seq, X_ent)
    tr, va, te = P.split_indices(y, flow_ids, X_ent, args.split, args.seed)
    logger.info("Split %s -> %d/%d/%d", args.split, len(tr), len(va), len(te))

    # cut points are fitted on the training partition and then compiled into
    # the switch as constants
    edges = (fit_edges(raw[tr]) if args.edges == "quantile" else FIXED_EDGES)
    F = apply_edges(raw, edges)
    logger.info("Switch feature ranges: %s",
                {n: (int(F[:, i].min()), int(F[:, i].max()))
                 for i, n in enumerate(FEATURE_NAMES)})
    occ = np.bincount(cell_index(F[tr]), minlength=KEY_SPACE)
    logger.info("Key space occupancy: %d of %d cells, %d with >=20 rows",
                int((occ > 0).sum()), KEY_SPACE, int((occ >= 20).sum()))

    weights = {c: len(y[tr]) / (args.num_classes * n)
               for c, n in zip(*np.unique(y[tr], return_counts=True))}

    if args.table_mode == "tree":
        tree = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=weights,
            random_state=args.seed,
        ).fit(F[tr], y[tr])
        grid = np.array(np.meshgrid(*[np.arange(KEY_RADIX)] * 5,
                                    indexing="ij")).reshape(5, -1).T
        proba = tree.predict_proba(grid)
        cell_cls, cell_conf = proba.argmax(axis=1), proba.max(axis=1)
        cell_support = np.full(KEY_SPACE, np.inf)
        extra = {"max_depth": args.max_depth,
                 "min_samples_leaf": args.min_samples_leaf,
                 "n_leaves": int(tree.get_n_leaves()),
                 "depth": int(tree.get_depth())}
        importance = dict(zip(FEATURE_NAMES,
                              tree.feature_importances_.round(4).tolist()))
    else:
        cell_cls, cell_conf, cell_support = empirical_table(
            F[tr], y[tr], args.num_classes, args.min_cell_support, weights)
        extra = {"min_cell_support": args.min_cell_support,
                 "cells_with_support": int((cell_support > 0).sum()),
                 "cells_above_support": int(
                     (cell_support >= args.min_cell_support).sum())}
        importance = None

    cell_action, cell_action_conf, policy_cls = action_confidence(
        F[tr], y[tr], cell_cls, args.min_cell_support)
    class_action = np.where(cell_cls == BENIGN_CLASS,
                            ACTION_ALLOW, ACTION_DROP)

    code_te = cell_index(F[te])
    pred = cell_cls[code_te]
    conf = cell_conf[code_te]
    resolved = conf >= args.confidence_floor

    acc = accuracy_score(y[te], pred)
    macro = f1_score(y[te], pred, average="macro", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y[te], pred, labels=range(args.num_classes), zero_division=0)

    logger.info("Data plane alone: acc=%.4f macro_f1=%.4f", acc, macro)
    logger.info("Resolved in switch: %.2f%% at conf>=%.2f, accuracy on "
                "those %.4f", 100 * resolved.mean(), args.confidence_floor,
                accuracy_score(y[te][resolved], pred[resolved])
                if resolved.any() else float("nan"))

    # The controller sees what the escalation report carries: the flow order
    # statistics plus the five buckets the switch already computed.
    from train_cnn import flow_summary
    ctrl = np.hstack([flow_summary(X_seq), F.astype(np.float32)])
    controller = build_controller(args.controller, args.seed, weights)
    controller.fit(ctrl[tr], y[tr])
    controller_pred = controller.predict(ctrl[te])
    ctrl_acc = accuracy_score(y[te], controller_pred)
    ctrl_macro = f1_score(y[te], controller_pred, average="macro",
                          zero_division=0)
    logger.info("Controller alone (%s): acc=%.4f macro_f1=%.4f",
                args.controller, ctrl_acc, ctrl_macro)

    composed = np.where(resolved, pred, controller_pred)
    comp_acc = accuracy_score(y[te], composed)
    comp_macro = f1_score(y[te], composed, average="macro", zero_division=0)
    logger.info("Composed system: acc=%.4f macro_f1=%.4f at %.1f%% resolved "
                "in switch", comp_acc, comp_macro, 100 * resolved.mean())

    floors = np.round(np.arange(0.30, 1.001, 0.05), 3)
    sweep = operating_points(conf, pred, y[te], controller_pred, floors)

    # The validation partition is spent on calibration rather than on model
    # selection: the table has no hyperparameters left to tune.
    code_va = cell_index(F[va])
    grid = np.round(np.arange(0.20, 1.0001, 0.01), 4)
    policy_points = drop_policy_points(
        cell_conf[code_va], class_action[code_va],
        cell_action_conf[code_va], cell_action[code_va], y[va],
        conf, class_action[code_te],
        cell_action_conf[code_te], cell_action[code_te], y[te],
        controller_pred, args.false_drop_budgets, args.delta, grid)
    for row in policy_points:
        flat, policy = row["label_confidence"], row["action_confidence"]
        logger.info("false-drop budget %.3f: flat-label covers %.1f%%; "
                    "policy-aligned covers %.1f%% at %.3f test error",
                    row["false_drop_budget"], 100 * flat["coverage"],
                    100 * policy["coverage"], policy["false_drop_rate"])

    deploy = next((r for r in policy_points
                   if abs(r["false_drop_budget"] -
                          args.deploy_false_drop) < 1e-12), None)
    if deploy is None or deploy["action_confidence"]["threshold"] is None:
        raise RuntimeError("the requested false-drop budget could not be "
                           "calibrated on this partition")
    deploy_policy = deploy["action_confidence"]
    deploy_tau = deploy_policy["threshold"]
    policy_entry_conf = np.where(cell_action == ACTION_DROP,
                                 cell_action_conf, 0.0)
    summary = export_entries(policy_cls, policy_entry_conf, deploy_tau,
                             args.num_classes, Path(args.out_dir))
    summary.update({
        "policy": "drop only; allow and uncertain cells escalate",
        "false_drop_budget": args.deploy_false_drop,
        "calibrated_threshold": deploy_tau,
    })
    export_bucketise(edges, Path(args.out_dir))
    logger.info("Exported %d entries of a %d-cell key space",
                summary["entries_installed"], summary["key_space"])

    results = {
        "model": f"dataplane_{args.table_mode}",
        "protocol": {
            "split": args.split,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "decision_packets": DECISION_PACKETS,
            "subsample_per_class": args.subsample_per_class,
            "corpus": args.data,
        },
        "hyperparameters": extra,
        "features": list(FEATURE_NAMES),
        "quantisation": {
            "mode": args.edges,
            "edges": {n: list(e) for n, e in zip(FEATURE_NAMES, edges)},
            "cells_occupied": int((occ > 0).sum()),
            "key_space": int(KEY_SPACE),
        },
        "feature_importance": importance,
        "test": {
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "per_class": {
                ID_TO_GROUP_LABEL.get(i, str(i)): {
                    "precision": float(prec[i]), "recall": float(rec[i]),
                    "f1": float(f1[i]), "support": int(sup[i])}
                for i in range(args.num_classes)},
        },
        "escalation": {
            "confidence_floor": args.confidence_floor,
            "resolved_fraction": float(resolved.mean()),
            "accuracy_on_resolved": float(
                accuracy_score(y[te][resolved], pred[resolved]))
            if resolved.any() else None,
            "escalated_fraction": float(1 - resolved.mean()),
        },
        "controller": {
            "model": args.controller,
            "features": "16 flow order statistics + 5 switch buckets",
            "accuracy": float(ctrl_acc),
            "macro_f1": float(ctrl_macro),
        },
        "composed": {
            "accuracy": float(comp_acc),
            "macro_f1": float(comp_macro),
            "resolved_fraction": float(resolved.mean()),
        },
        "operating_points": sweep,
        "policy_comparison": {
            "actions": {"0": "allow", "1": "drop"},
            "fast_path": "drop only; allow and uncertain decisions defer",
            "points": policy_points,
        },
        "deployment_policy": {
            "false_drop_budget": args.deploy_false_drop,
            "threshold": deploy_tau,
            "coverage": deploy_policy["coverage"],
            "false_drop_rate": deploy_policy["false_drop_rate"],
            "action_accuracy": deploy_policy["action_accuracy"],
            "entries_installed": summary["entries_installed"],
        },
        "table": summary,
        "confusion_matrix": confusion_matrix(
            y[te], composed, labels=range(args.num_classes)).tolist(),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", args.results)


if __name__ == "__main__":
    main()
