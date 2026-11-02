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
from scipy.stats import beta as beta_dist, mannwhitneyu
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
    observed positions. As in the P4 action, an average that is still zero
    is re-seeded by the next observation, so a zero first gap does not
    freeze the switch and the host into different states.
    """
    n, length = values.shape
    out = np.zeros(n, dtype=np.int64)
    for t in range(length):
        v = values[:, t]
        m = valid[:, t]
        seed = m & (out == 0)
        upd = m & (out != 0)
        out[seed] = v[seed]
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


EXTRA_REPORT_FEATURES = ("fwd_pkts", "fwd_bytes", "iat_min_us", "iat_max_us",
                         "duration_us", "len_dev", "first_len")


def report_registers(X_seq: np.ndarray, X_ent: np.ndarray) -> np.ndarray:
    """
    The twelve registers the escalation report carries at packet ten: the
    five policy registers, then the seven in ``EXTRA_REPORT_FEATURES``,
    with the integer recurrences of feature_extraction.p4.
    """
    base = switch_raw(X_seq, X_ent)
    X = X_seq[:, :DECISION_PACKETS]
    L = np.rint(np.expm1(X[:, :, 0])).astype(np.int64)
    G = np.rint(np.expm1(X[:, :, 1]) * 1000.0).astype(np.int64)[:, 1:]
    D = X[:, :, 2] > 0.5                 # 1 = direction of the first packet
    dev = np.zeros(len(L), np.int64)
    ewma = np.zeros(len(L), np.int64)
    for t in range(L.shape[1]):
        v = L[:, t]
        d = np.abs(v - ewma)
        if t:
            dev = np.where(dev == 0, d, dev -
                           (dev >> EWMA_SHIFT) + (d >> EWMA_SHIFT))
        ewma = np.where(ewma == 0, v, ewma -
                        (ewma >> EWMA_SHIFT) + (v >> EWMA_SHIFT))
    extra = np.stack([D.sum(1), (L * D).sum(1), G.min(1), G.max(1), G.sum(1),
                      dev, L[:, 0]], axis=1)
    return np.hstack([base, extra.astype(np.float64)])


FIXED_EDGES = (BYTE_EDGES, LEN_EDGES, IAT_EDGES,
               MINLEN_EDGES, MAXLEN_EDGES)


def fit_edges(raw_tr: np.ndarray, n_buckets: int = None):
    """
    Equal-frequency cut points, read off the training partition alone.
    ``n_buckets`` defaults to the key resolution in force (``KEY_RADIX``).

    The hand-picked edges leave the table almost empty: they place 98% of
    flows in one source-diversity bucket and use three of six count
    buckets, so the switch discards resolution the table could have held.
    Quantile edges spread the corpus across the key space. They are
    constants compiled into the P4 program, so this costs the switch
    nothing at run time.
    """
    qs = np.linspace(0, 1, (n_buckets or KEY_RADIX) + 1)[1:-1]
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
                   out_dir: Path, actions: np.ndarray = None) -> dict:
    """
    Emit one match-action entry per cell whose score clears the floor.
    Every key field takes KEY_RADIX values, so the key space is enumerable
    and each entry is an exact ternary rule; no interval-to-mask
    approximation is involved. ``actions`` names each cell's fast-path
    action (1 drop, 2 rate-limit, 3 redirect); the default is drop.
    """
    grid = np.array(np.meshgrid(*[np.arange(KEY_RADIX)] * 5, indexing="ij"))
    keys = grid.reshape(5, -1).T
    if actions is None:
        actions = np.full(len(keys), ACTION_DROP)
    keep = conf >= confidence_floor
    lines = ["# table_add <table> <action> <key> => <args> [priority]",
             "# key fields: " + " ".join(FEATURE_NAMES),
             f"# levels: {KEY_RADIX}"]
    for k, c, q, a in zip(keys[keep], cls[keep], conf[keep], actions[keep]):
        key = " ".join(f"{int(v)}&&&0xFF" for v in k)
        lines.append("table_add SplitMambaIngress.flow_classifier set_verdict "
                     f"{key} => {int(c)} {int(round(q * 255))} {int(a)} 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classifier_entries.txt").write_text("\n".join(lines) + "\n")

    summary = {
        "key_space": int(len(keys)),
        "entries_installed": int(keep.sum()),
        "entries_defaulted_to_escalation": int((~keep).sum()),
        "confidence_floor": confidence_floor,
        "class_distribution": np.bincount(
            cls[keep], minlength=n_classes).tolist(),
        "action_distribution": {str(a): int((actions[keep] == a).sum())
                                for a in np.unique(actions[keep])},
    }
    (out_dir / "classifier_entries.json").write_text(
        json.dumps(summary, indent=2))
    return summary


MEASURED_FEATURES = ("pkt_count", "byte_count", "mean_iat_us",
                     "dstport_fanout", "src_diversity")


# Levels per key field. Chosen among 6, 8, 10 and 12 on development seeds
# 0 to 29; reported results use seeds 30 to 59.
KEY_RADIX = 10
KEY_SPACE = KEY_RADIX ** 5


def set_key_levels(levels: int) -> None:
    global KEY_RADIX, KEY_SPACE
    KEY_RADIX, KEY_SPACE = int(levels), int(levels) ** 5


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
            "true_sources_quartiles": [float(q) for q in
                                       np.percentile(t, [25, 50, 75])],
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


def aggregate_measured(paths, n_seeds, num_classes):
    """The measured-feature analysis over several replays and flow splits."""
    reps = []
    for p in paths:
        runs = [evaluate_measured(Path(p), s, num_classes)
                for s in range(n_seeds)]
        first = runs[0]
        sep = first["ddos_dos_separation"]
        reps.append({
            "source": p, "n_flows": first["n_flows"],
            "estimator_vs_truth_pearson_r": first["estimator_vs_truth_pearson_r"],
            "auc_truth": sep["ground_truth"]["auc"],
            "auc_truth_ci95": sep["ground_truth"]["auc_ci95"],
            "auc_switch": sep["switch_estimate"]["auc"],
            "auc_switch_ci95": sep["switch_estimate"]["auc_ci95"],
            "by_class": first["src_diversity_by_class"],
            "diversity_gain_pp": float(np.mean(
                [r["diversity_gain_pp"] for r in runs])),
            "beyond_class_marginal_pp": float(np.mean(
                [r["diversity_gain_beyond_class_marginal_pp"] for r in runs])),
            "per_seed": [{"gain": r["diversity_gain_pp"], "beyond":
                          r["diversity_gain_beyond_class_marginal_pp"]}
                         for r in runs]})

    def span(key):
        v = np.array([r[key] for r in reps], dtype=float)
        return {"mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max())}
    return {"replays": reps, "split_seeds_per_replay": n_seeds,
            "summary": {k: span(k) for k in (
                "estimator_vs_truth_pearson_r", "auc_truth", "auc_switch",
                "diversity_gain_pp", "beyond_class_marginal_pp")}}


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
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


BOOT_BLOCKS = 50
BOOT_RESAMPLES = 400


def block_bootstrap_upper(err_in_order, delta, n_blocks=BOOT_BLOCKS,
                          n_boot=BOOT_RESAMPLES, seed=0):
    """
    Upper (1 - delta) quantile of the selective error when contiguous
    blocks of calibration flows, in extraction order, are resampled.
    Flows of one capture sit next to each other, so a block keeps their
    dependence together instead of counting them as independent trials.
    """
    blocks = np.array_split(err_in_order, min(n_blocks, len(err_in_order)))
    k = np.array([b.sum() for b in blocks], dtype=float)
    n = np.array([len(b) for b in blocks], dtype=float)
    pick = np.random.RandomState(seed).randint(0, len(blocks),
                                               (n_boot, len(blocks)))
    return float(np.quantile(k[pick].sum(1) / n[pick].sum(1), 1 - delta))


# The dependence-aware walk starts where the strictest studied target could
# first be certified: the smallest admitted count whose exact bound with no
# error is within this target. The count uses no labels.
WALK_START_TARGET = 0.005


def walk_start(delta, target=WALK_START_TARGET):
    """Smallest n with clopper_pearson_upper(0, n, delta) <= target."""
    return int(np.ceil(np.log(delta) / np.log1p(-target)))


def cluster_weights(groups, n_boot=BOOT_RESAMPLES, seed=0):
    """Group index of every flow and multinomial resampling counts per group."""
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    gidx = gidx.ravel()
    G = int(gidx.max()) + 1
    pick = np.random.RandomState(seed).randint(0, G, (n_boot, G))
    W = np.stack([np.bincount(p, minlength=G) for p in pick]).astype(float)
    return gidx, W


def cluster_bootstrap_curve(conf, wrong, grid, clusters, delta):
    """
    Upper (1 - delta) quantile of the selective error at every threshold of
    ``grid`` when whole groups of calibration flows are resampled. The groups
    are the units the partition assigns, so they move between partitions
    together and are the independent units of the calibration sample.
    """
    gidx, W = clusters
    grid = np.asarray(grid, dtype=float)
    asc = np.argsort(grid)
    passed = np.searchsorted(grid[asc], conf, side="right")
    K = np.zeros((len(grid), W.shape[1]))
    N = np.zeros_like(K)
    ok = passed > 0
    np.add.at(K, (passed[ok] - 1, gidx[ok]), wrong[ok].astype(float))
    np.add.at(N, (passed[ok] - 1, gidx[ok]), 1.0)
    # Row t holds the flows that pass threshold t in ascending order.
    K = np.cumsum(K[::-1], axis=0)[::-1]
    N = np.cumsum(N[::-1], axis=0)[::-1]
    kb, nb = K @ W.T, N @ W.T
    upper = np.quantile(np.where(nb > 0, kb / np.maximum(nb, 1.0), 0.0),
                        1 - delta, axis=1)
    out = np.empty(len(grid))
    out[asc] = upper
    return out


def threshold_bounds(conf_cal, pred_cal, y_cal, delta, grid, order=None,
                     groups=None, rule="dependence"):
    """
    Walk-order list of (threshold, admitted count, upper bound), computed
    once so that every target reuses it.

    Without ``order`` the bound is the exact binomial one, valid for
    exchangeable calibration flows. CICIoT2023 flows within one capture are
    dependent, so with ``order`` (calibration flows in extraction order) the
    default ``dependence`` rule takes the largest of three bounds: the exact
    binomial bound, a block bootstrap quantile over neighbouring flows, and,
    when ``groups`` gives the units the partition assigns (labels, or the
    output of cluster_weights), a cluster bootstrap quantile over those
    units. ``rule="block"`` is the earlier rule, the block bound alone.
    """
    wrong = pred_cal != y_cal
    grid = sorted(grid, reverse=True)
    dependence = order is not None and rule == "dependence"
    grp = None
    if dependence and groups is not None:
        clusters = groups if isinstance(groups, tuple) else \
            cluster_weights(groups)
        grp = cluster_bootstrap_curve(conf_cal, wrong, grid, clusters, delta)
    rows = []
    for i, tau in enumerate(grid):
        sel = conf_cal >= tau
        n = int(sel.sum())
        if n == 0:
            rows.append((float(tau), 0, 1.0))
            continue
        exact = clopper_pearson_upper(int(wrong[sel].sum()), n, delta)
        if order is None:
            upper = exact
        else:
            upper = block_bootstrap_upper(wrong[order][sel[order]], delta)
            if dependence:
                upper = max(upper, exact)
                if grp is not None:
                    upper = max(upper, float(grp[i]))
        rows.append((float(tau), n, float(upper)))
    return rows


def walk_thresholds(rows, epsilon, delta, n_start=None):
    """
    Fixed-sequence walk over threshold_bounds rows: accept thresholds while
    the bound stays within ``epsilon`` and stop at the first failure. Testing
    starts at the first threshold admitting ``n_start`` flows, or, without
    it, enough flows to meet ``epsilon`` with no error. Skipping on n alone
    uses no labels, so the fixed sequence stays valid.
    """
    chosen, certified = None, None
    for tau, n, upper in rows:
        if n_start is not None:
            if n < n_start:
                continue
        elif clopper_pearson_upper(0, n, delta) > epsilon:
            continue
        if upper <= epsilon:
            chosen, certified = tau, upper
        else:
            break
    return chosen, certified


def calibrate_threshold(conf_cal, pred_cal, y_cal, epsilon, delta, grid,
                        order=None, groups=None, rule="dependence"):
    """
    Lowest threshold whose calibration bound is below ``epsilon``.

    Thresholds are tested in a pre-specified descending order, which walks
    from the most conservative setting towards more coverage, and testing
    stops at the first threshold that fails (see threshold_bounds for the
    bounds). The dependence rule starts every walk at the same label-free
    point, the first threshold admitting walk_start() flows, so a larger
    target never stops earlier than a smaller one; the binomial and the
    earlier block rule start at the first threshold that could meet
    ``epsilon`` with no error. None of these is a finite-sample guarantee
    for dependent flows; the selected threshold is audited on a held-out
    partition.
    """
    rows = threshold_bounds(conf_cal, pred_cal, y_cal, delta, grid, order,
                            groups, rule)
    dependence = order is not None and rule == "dependence"
    return walk_thresholds(rows, epsilon, delta,
                           walk_start(delta) if dependence else None)


# Candidate rules compared on development seeds before the dependence rule
# was fixed: which bounds are combined by their maximum, and where the walk
# starts ("target": first count that could meet the target with no error;
# "common": walk_start() for every target).
CANDIDATE_BOUNDS = {"cp": ("cp",), "blk": ("blk",), "blk_cp": ("blk", "cp"),
                    "grp": ("grp",), "blk_grp": ("blk", "grp"),
                    "all": ("blk", "grp", "cp")}


def candidate_outcomes(s_va, a_va, t_va, s_te, a_te, t_te, budgets, delta,
                       grid, order, groups):
    """Test coverage, error and target met for every candidate rule."""
    wrong = a_va != t_va
    grid = sorted(grid, reverse=True)
    grp = (cluster_bootstrap_curve(s_va, wrong, grid, groups, delta)
           if groups is not None else None)
    parts = []
    for i, tau in enumerate(grid):
        sel = s_va >= tau
        n = int(sel.sum())
        if n == 0:
            parts.append((float(tau), 0, {"cp": 1.0, "blk": 1.0, "grp": 1.0}))
            continue
        b = block_bootstrap_upper(wrong[order][sel[order]], delta)
        parts.append((float(tau), n, {
            "cp": clopper_pearson_upper(int(wrong[sel].sum()), n, delta),
            "blk": b, "grp": float(grp[i]) if grp is not None else b}))
    out = {}
    for eps in budgets:
        res = {}
        for name, comps in CANDIDATE_BOUNDS.items():
            rows = [(t, n, max(u[c] for c in comps)) for t, n, u in parts]
            for start, n0 in (("target", None), ("common", walk_start(delta))):
                tau, _ = walk_thresholds(rows, eps, delta, n0)
                sel = (s_te >= tau) if tau is not None else \
                    np.zeros(len(s_te), dtype=bool)
                err = (float((a_te[sel] != t_te[sel]).mean())
                       if sel.any() else None)
                res[f"{name}|{start}"] = [float(sel.mean()), err,
                                          err is None or err <= eps]
        out[str(float(eps))] = res
    return out


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


# Fine labels in index order: Benign DDoS DoS Mirai Recon Web Spoofing
# BruteForce. Actions: 0 allow, 1 drop, 2 rate-limit, 3 redirect for inspection.
ACTION_MAPS = {
    "2-action": np.array([0, 1, 1, 1, 1, 1, 1, 1]),
    "3-action": np.array([0, 1, 1, 1, 2, 2, 1, 2]),
    "4-action": np.array([0, 1, 1, 1, 2, 3, 3, 2]),
}
# The two-action drops, except that spoofing goes to inspection: spoofing
# captures record the victims' own connections under the attack label.
SPOOF_REDIRECT_MAP = np.array([0, 1, 1, 1, 1, 1, 3, 1])
# Floods are dropped; the four application attacks, which the registers
# confuse with each other at the tenth packet, share one milder action.
GROUPED_MAPS = {"flood-inspect": np.array([0, 1, 1, 1, 3, 3, 3, 3]),
                "flood-ratelimit": np.array([0, 1, 1, 1, 2, 2, 2, 2])}
TIERED_MAPS = {"3-action": ACTION_MAPS["3-action"],
               "4-action": ACTION_MAPS["4-action"],
               "spoof-redirect": SPOOF_REDIRECT_MAP,
               "flood-inspect": GROUPED_MAPS["flood-inspect"]}
# Irreversible actions get the strictest budget.
TIERED_BUDGETS = {1: 0.01, 2: 0.05, 3: 0.10}
# Attack classes whose flows are ordinary-looking connections, not floods.
APPLICATION_ATTACKS = tuple(i for i, n in ID_TO_GROUP_LABEL.items()
                            if n in ("Recon", "Web", "Spoofing", "BruteForce"))
STUDY_SPLITS = ("flow-isolated", "blocked", "context-disjoint")
STUDY_GRID = np.round(np.arange(0.20, 1.0001, 0.01), 4)


def cell_gates(counts, leaf, leaf_conf, amap):
    """Admission score (-1 when ineligible) and assigned action per cell."""
    support = counts.sum(axis=1)
    pooled = np.zeros((len(counts), int(amap.max()) + 1))
    for c, a in enumerate(amap):
        pooled[:, a] += counts[:, c]
    best = pooled.argmax(axis=1)
    attack = support - counts[:, BENIGN_CLASS]
    attack_major = attack >= counts[:, BENIGN_CLASS]
    attack_conf = wilson_lower(attack, support)
    dominant = counts[:, 1:].argmax(axis=1) + 1
    leaf_attack = leaf != BENIGN_CLASS
    # Hierarchical selective classification over the flat taxonomy: an
    # unconfident attack leaf climbs to the attack node, which can act only
    # when every attack family shares one action.
    one_action = len(set(amap[1:].tolist())) == 1
    hsc = np.maximum(leaf_conf, attack_conf) if one_action else leaf_conf
    gates = {
        "label": (np.where(leaf_attack, leaf_conf, -1.0), amap[leaf]),
        "action": (np.where(best != ACTION_ALLOW,
                            wilson_lower(pooled.max(axis=1), support), -1.0),
                   best),
        "binary": (np.where(attack_major, attack_conf, -1.0), amap[dominant]),
        "hsc": (np.where(leaf_attack, hsc, -1.0), amap[leaf]),
    }
    if one_action:
        prob = np.divide(attack, support, out=np.zeros_like(attack),
                         where=support > 0)
        gates["probability"] = (np.where(attack_major, prob, -1.0),
                                np.full(len(counts), amap[1]))
    return gates


def aggregation_mass(counts, amap):
    """
    Share of training flows that pooling by action adds to the top label:
    sum over cells of (largest action count - largest label count) / N.
    It is the within-action confusion the action gate can exploit, and is
    zero when every confusion crosses an action boundary.
    """
    pooled = np.zeros((len(counts), int(amap.max()) + 1))
    for c, a in enumerate(amap):
        pooled[:, a] += counts[:, c]
    return float((pooled.max(axis=1) - counts.max(axis=1)).sum() / counts.sum())


def confusion_structure(counts, amap, names=None, top_k=8):
    """
    How the training flows that are not of their cell's top label relate to
    it: the share that shares the top label's action (pooling can only help
    there), and the largest (top label, other label) masses.
    """
    top = counts.argmax(axis=1)
    total = within = 0.0
    pairs = {}
    for c in range(counts.shape[1]):
        off = top != c
        total += counts[off, c].sum()
        within += counts[off & (amap[top] == amap[c]), c].sum()
        for t in np.unique(top[off & (counts[:, c] > 0)]):
            pairs[(int(t), c)] = float(counts[off & (top == t), c].sum())
    lead = sorted(pairs.items(), key=lambda kv: -kv[1])[:top_k]
    label = (lambda i: names[i]) if names else (lambda i: int(i))
    return {"within_action_share": float(within / total) if total else None,
            "leading_pairs": [{"top": label(t), "other": label(c),
                               "flows": v, "same_action": bool(amap[t] == amap[c])}
                              for (t, c), v in lead]}


def random_action_maps(n_classes, n_actions, n_maps, seed=0):
    """
    Distinct action maps that allow class 0 and split the attack classes
    into ``n_actions`` non-empty groups. Actions share one budget here, so
    two maps that differ only by renaming actions are the same map.
    """
    rng = np.random.RandomState(seed)
    seen, maps = set(), []
    for _ in range(200 * n_maps):
        if len(maps) == n_maps:
            break
        m = rng.randint(0, n_actions, n_classes - 1).tolist()
        if len(set(m)) < n_actions:
            continue
        names = {}
        canon = tuple(names.setdefault(v, len(names)) + 1 for v in m)
        if canon not in seen:
            seen.add(canon)
            maps.append(np.array((0,) + canon))
    return maps


def map_gains(counts, leaf, leaf_conf, c_va, c_te, y_va, y_te, maps, eps,
              delta, order, clusters):
    """
    Action and label gate coverage at one target under the dependence rule
    for each action map, with two training-set predictors of the gain: the
    aggregation mass, and the crossing mass, the share of training flows in
    cells whose action score reaches 1 - eps while their label score does
    not, which are the cells only pooling can admit at that target.
    """
    out = []
    support = counts.sum(axis=1)
    for amap in maps:
        gates = cell_gates(counts, leaf, leaf_conf, amap)
        t_va, t_te = amap[y_va], amap[y_te]
        cross = (gates["action"][0] >= 1 - eps) & (gates["label"][0] < 1 - eps)
        row = {"map": amap.tolist(),
               "aggregation_mass": aggregation_mass(counts, amap),
               "crossing_mass": float(support[cross].sum() / support.sum())}
        for g in ("label", "action"):
            score, act = gates[g]
            tau, _ = walk_thresholds(
                threshold_bounds(score[c_va], act[c_va], t_va, delta,
                                 STUDY_GRID, order, clusters),
                eps, delta, walk_start(delta))
            sel = (score[c_te] >= tau) if tau is not None else \
                np.zeros(len(c_te), dtype=bool)
            row[f"{g}_cov"] = float(sel.mean())
            row[f"{g}_err"] = (float((act[c_te][sel] != t_te[sel]).mean())
                               if sel.any() else None)
            # Best test coverage within the target: what any threshold of
            # this gate could reach.
            s_te, a_te = score[c_te], act[c_te]
            front = 0.0
            for t in STUDY_GRID:
                s = s_te >= t
                if s.any() and (a_te[s] != t_te[s]).mean() <= eps:
                    front = max(front, float(s.mean()))
            row[f"{g}_front"] = front
        out.append(row)
    return out


def summarise_maps(per_seed, eps, n_boot=2000):
    """Per map: mean gain with a seed bootstrap interval; across maps: rank
    correlation between aggregation mass and gain."""
    from scipy.stats import spearmanr
    rng = np.random.RandomState(0)
    maps = []
    for i, first in enumerate(per_seed[0]["maps"]):
        d = 100 * np.array([r["maps"][i]["action_cov"] - r["maps"][i]["label_cov"]
                            for r in per_seed])
        boot = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(axis=1)
        miss = sum(r["maps"][i]["action_err"] is not None
                   and r["maps"][i]["action_err"] > eps for r in per_seed)
        maps.append({"map": first["map"],
                     "actions": int(max(first["map"])),
                     "aggregation_mass": float(np.mean(
                         [r["maps"][i]["aggregation_mass"] for r in per_seed])),
                     "crossing_mass": float(np.mean(
                         [r["maps"][i]["crossing_mass"] for r in per_seed])),
                     "action_front": float(np.mean(
                         [r["maps"][i]["action_front"] for r in per_seed])),
                     "gain_points": float(d.mean()),
                     "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                     "action_misses": int(miss), "n_seeds": len(d)})
    mass = np.array([m["aggregation_mass"] for m in maps])
    cross = np.array([m["crossing_mass"] for m in maps])
    gain = np.array([m["gain_points"] for m in maps])
    rho, p = spearmanr(mass, gain)
    rho_c, p_c = spearmanr(cross, gain)
    multi = np.array([m["actions"] > 1 for m in maps])
    rho_m, p_m = spearmanr(cross[multi], gain[multi])
    slope = np.polyfit(100 * cross, gain, 1)
    return {"maps": maps, "spearman_rho": float(rho), "spearman_p": float(p),
            "crossing_rho": float(rho_c), "crossing_p": float(p_c),
            "crossing_rho_multi_action": float(rho_m),
            "crossing_p_multi_action": float(p_m),
            "slope_points_per_point_of_crossing": float(slope[0]),
            "intercept_points": float(slope[1]),
            "positive_gain_maps": int((gain > 0).sum()),
            "ci_above_zero_maps": int(sum(m["ci95"][0] > 0 for m in maps)),
            "n_maps": len(maps)}


def map_study(args):
    """
    How the action gate's gain depends on the action map: random maps that
    split the seven attack classes into two or three actions, together with
    the maps of the main study, at one target on the flow-isolated rule.
    """
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    maps = ([ACTION_MAPS[m] for m in ACTION_MAPS]
            + random_action_maps(args.num_classes, 2, args.map_count, 1)
            + random_action_maps(args.num_classes, 3, args.map_count, 2)
            + [np.arange(args.num_classes)])
    uniq, seen = [], set()
    for m in maps:
        names = {}
        key = tuple(names.setdefault(v, len(names)) for v in m.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    eps = args.map_target
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    per_seed = []
    for seed in seeds:
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        raw = switch_raw(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, "flow-isolated", seed)
        order = np.argsort(va, kind="stable")
        clusters = cluster_weights(fid[va])
        F = apply_edges(raw, fit_edges(raw[tr]))
        c_tr, c_va, c_te = (cell_index(F[i]) for i in (tr, va, te))
        weights = {c: len(tr) / (args.num_classes * n) for c, n in
                   zip(*np.unique(y[tr], return_counts=True))}
        leaf, leaf_conf, _ = empirical_table(F[tr], y[tr], args.num_classes,
                                             0, weights)
        counts = np.zeros((KEY_SPACE, args.num_classes))
        np.add.at(counts, (c_tr, y[tr]), 1.0)
        per_seed.append({"seed": seed, "maps": map_gains(
            counts, leaf, leaf_conf, c_va, c_te, y[va], y[te], uniq, eps,
            args.delta, order, clusters)})
        logger.info("map study seed %d done (%d maps)", seed, len(uniq))
    out = {"config": {"split": "flow-isolated", "target": eps,
                      "delta": args.delta, "seeds": list(seeds),
                      "key_levels": KEY_RADIX,
                      "random_maps_per_size": args.map_count,
                      "calibration": "dependence rule"},
           "summary": summarise_maps(per_seed, eps), "per_seed": per_seed}
    Path(args.map_study_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.map_study_out).write_text(json.dumps(out))
    logger.info("Map study written to %s: rho %.2f (aggregation), %.2f "
                "(crossing)", args.map_study_out,
                out["summary"]["spearman_rho"], out["summary"]["crossing_rho"])


def gate_outcomes(s_va, a_va, t_va, s_te, a_te, t_te, budgets, delta,
                  grid=STUDY_GRID, order=None, groups=None, conn_te=None):
    """
    Calibrated coverage under the exact binomial bound (cal), the earlier
    block rule (blk) and the dependence rule (dep), with the plug-in
    threshold and the test-set frontier for reference.
    """
    def at(tau):
        if tau is None:
            return 0.0, None
        sel = s_te >= tau
        if not sel.any():
            return 0.0, None
        return float(sel.mean()), float((a_te[sel] != t_te[sel]).mean())

    curve = []
    for tau in grid:
        cov, err = at(tau)
        if err is not None:
            curve.append((float(tau), cov, err))
    val_err = []
    for tau in grid:
        sel = s_va >= tau
        if sel.any():
            val_err.append(
                (float(tau), float((a_va[sel] != t_va[sel]).mean())))
    rows = {"cal": threshold_bounds(s_va, a_va, t_va, delta, grid)}
    if order is not None:
        rows["blk"] = threshold_bounds(s_va, a_va, t_va, delta, grid, order,
                                       rule="block")
        rows["dep"] = threshold_bounds(s_va, a_va, t_va, delta, grid, order,
                                       groups)
    out = {}
    for eps in budgets:
        tau, _ = walk_thresholds(rows["cal"], eps, delta)
        cov, err = at(tau)
        plug = min([t for t, e in val_err if e <= eps], default=None)
        pcov, perr = at(plug)
        out[float(eps)] = {
            "cal_cov": cov, "cal_err": err,
            "cal_met": err is None or err <= eps,
            "plug_cov": pcov, "plug_err": perr,
            "plug_met": perr is None or perr <= eps,
            "front_cov": max([c for _, c, e in curve if e <= eps], default=0.0),
        }
        if order is not None:
            for kind, start in (("blk", None), ("dep", walk_start(delta))):
                btau, _ = walk_thresholds(rows[kind], eps, delta, start)
                bcov, berr = at(btau)
                sv = (s_va >= btau) if btau is not None else \
                    np.zeros(len(s_va), dtype=bool)
                st = (s_te >= btau) if btau is not None else \
                    np.zeros(len(s_te), dtype=bool)
                out[float(eps)].update({
                    f"{kind}_cov": bcov, f"{kind}_err": berr,
                    f"{kind}_met": berr is None or berr <= eps,
                    f"{kind}_n_val": int(sv.sum()),
                    f"{kind}_k_val": int((a_va[sv] != t_va[sv]).sum()),
                    f"{kind}_n_te": int(st.sum()),
                    f"{kind}_k_te": int((a_te[st] != t_te[st]).sum())})
                if conn_te is not None:
                    # The same error counted per connection: a 5-tuple is
                    # wrong once if any of its admitted records is.
                    adm = np.unique(conn_te[st])
                    bad, per = np.unique(conn_te[st & (a_te != t_te)],
                                         return_counts=True)
                    cerr = len(bad) / len(adm) if len(adm) else None
                    out[float(eps)].update({
                        f"{kind}_conn_err": cerr,
                        f"{kind}_conn_met": cerr is None or cerr <= eps,
                        f"{kind}_k_te_top": int(per.max()) if len(per) else 0})
    return out, curve


def tiered_outcomes(s_va, a_va, t_va, s_te, a_te, t_te, delta, order=None,
                    groups=None, rule="dependence", conn_te=None, cells=None):
    """
    One validation threshold per action, each at its own budget. ``cells``
    is the per-cell (score, action) pair of the gate, used to count the
    entries each action would install.
    """
    total = np.zeros(len(s_te), dtype=bool)
    per_action = {}
    dependence = order is not None and rule == "dependence"
    benign_te = t_te == ACTION_ALLOW
    for a, eps in TIERED_BUDGETS.items():
        if not (a_va == a).any():
            continue
        masked = np.where(a_va == a, s_va, -1.0)
        rows = threshold_bounds(masked, a_va, t_va, delta, STUDY_GRID, order,
                                groups, rule)
        tau, _ = walk_thresholds(rows, eps, delta,
                                 walk_start(delta) if dependence else None)
        sel = ((a_te == a) & (s_te >= tau)) if tau is not None else \
            np.zeros(len(s_te), dtype=bool)
        total |= sel
        err = float((t_te[sel] != a).mean()) if sel.any() else None
        per_action[a] = {"budget": eps, "coverage": float(sel.mean()),
                         "error": err, "met": err is None or err <= eps,
                         "threshold": tau, "certified": tau is not None,
                         "admitted": int(sel.sum()),
                         "benign_admitted": int((sel & benign_te).sum()),
                         # Share of all benign test flows given this action.
                         "benign_rate": float(sel[benign_te].mean())
                         if benign_te.any() else None}
        if cells is not None:
            c_score, c_act = cells
            per_action[a]["entries"] = int(
                ((c_act == a) & (c_score >= tau)).sum()) if tau is not None else 0
        if dependence:
            # The first set the walk tests: why an action is never certified.
            first = next((r for r in rows if r[1] >= walk_start(delta)), None)
            if first is not None:
                s0 = masked >= first[0]
                bad = t_va[s0][t_va[s0] != a]
                per_action[a]["start"] = {
                    "n": int(s0.sum()), "bound": first[2],
                    "error": float((t_va[s0] != a).mean()),
                    "wrong_by_true_action": np.bincount(
                        bad, minlength=len(TIERED_BUDGETS) + 1).tolist()}
        if conn_te is not None:
            adm = np.unique(conn_te[sel])
            bad = np.unique(conn_te[sel & (t_te != a)])
            cerr = len(bad) / len(adm) if len(adm) else None
            per_action[a].update({"conn_error": cerr,
                                  "conn_met": cerr is None or cerr <= eps})
    err = float((a_te[total] != t_te[total]).mean()) if total.any() else None
    return {"coverage": float(total.mean()), "error": err,
            "per_action": per_action}


def export_tiered(F, y, tr, va, te, leaf, leaf_conf, amap, edges, args,
                  out_dir, groups=None):
    """
    Compile a multi-action policy: each action gets its own budget
    (TIERED_BUDGETS) and its own dependence-calibrated threshold, and every
    admitted cell becomes an entry naming that action. ``groups`` are the
    partition units of the validation flows.
    """
    c_tr, c_va, c_te = (cell_index(F[i]) for i in (tr, va, te))
    counts = np.zeros((KEY_SPACE, args.num_classes))
    np.add.at(counts, (c_tr, y[tr]), 1.0)
    score, act = cell_gates(counts, leaf, leaf_conf, amap)["action"]
    order = np.argsort(va, kind="stable")
    clusters = cluster_weights(groups) if groups is not None else None
    rule = "block" if args.deploy_calibration == "block" else "dependence"
    t_va, t_te = amap[y[va]], amap[y[te]]
    # The P4 guard defers entries whose confidence byte is under 179.
    guard = np.rint(np.clip(score, 0, 1) * 255) >= 179
    admitted = np.zeros(KEY_SPACE, dtype=bool)
    per_action = {}
    for a, eps in TIERED_BUDGETS.items():
        mine = (act == a) & guard
        tau, ub = calibrate_threshold(np.where(mine[c_va], score[c_va], -1.0),
                                      act[c_va], t_va, eps, args.delta,
                                      STUDY_GRID, order, clusters, rule)
        cells = mine & (score >= tau) if tau is not None else \
            np.zeros(KEY_SPACE, dtype=bool)
        admitted |= cells
        sel = cells[c_te]
        per_action[str(a)] = {
            "budget": eps, "threshold": tau, "calibration_upper_bound": ub,
            "entries": int(cells.sum()), "coverage": float(sel.mean()),
            "error": float((t_te[sel] != a).mean()) if sel.any() else None}
    # Report the dominant fine class among those the action covers.
    member = amap[None, :] == act[:, None]
    cls = np.where(member, counts, -1).argmax(axis=1)
    conf = np.where(admitted, score, 0.0)
    summary = export_entries(cls, conf, 1e-9, args.num_classes, out_dir,
                             actions=act)
    export_bucketise(edges, out_dir)
    sel = admitted[c_te]
    out = {"action_map": amap.tolist(), "budgets": TIERED_BUDGETS,
           "per_action": per_action, "entries": summary["entries_installed"],
           "coverage": float(sel.mean()),
           "error": float((act[c_te][sel] != t_te[sel]).mean())
           if sel.any() else None}
    logger.info("tiered %s: %s", amap.tolist(), per_action)
    return out


def _per_action_summary(per_action_rows):
    """
    How often each action is certified across seeds, and its mean entries,
    coverage, selective error and benign rate where it is. Rows written
    before these fields existed contribute only to the miss counts.
    """
    out = {"certified_seeds_by_action": {}, "entries_mean_by_action": {},
           "coverage_mean_by_action": {}, "error_mean_by_action": {},
           "benign_rate_mean_by_action": {}}
    for a in TIERED_BUDGETS:
        rows = [pa.get(a) or pa.get(str(a)) for pa in per_action_rows]
        rows = [r for r in rows if r is not None]
        cert = [r for r in rows if r.get("certified", r["coverage"] > 0)]
        out["certified_seeds_by_action"][str(a)] = len(cert)
        for key, field in (("entries_mean_by_action", "entries"),
                           ("coverage_mean_by_action", "coverage"),
                           ("error_mean_by_action", "error"),
                           ("benign_rate_mean_by_action", "benign_rate")):
            vals = [r[field] for r in cert
                    if r.get(field) is not None]
            out[key][str(a)] = float(np.mean(vals)) if vals else None
    return out


def summarise_study(per_seed, budgets, n_boot=2000):
    """Paired seed-level contrasts, with bootstrap intervals and misses."""
    from scipy.stats import wilcoxon
    rng = np.random.RandomState(0)
    pairs = [("action", "label"), ("action", "binary"), ("action", "hsc"),
             ("action", "probability")]
    summary = {}
    splits = sorted({r["split"] for r in per_seed})
    for split in splits:
        rows = [r for r in per_seed if r["split"] == split]
        summary[split] = {}
        for amap in rows[0]["gates"]:
            gates = rows[0]["gates"][amap]
            block = {"gates": {}, "contrasts": {}}
            for g in gates:
                block["gates"][g] = {}
                for eps in budgets:
                    key = str(float(eps))
                    vals = [r["gates"][amap][g][key] for r in rows]
                    cov = np.array([v["cal_cov"] for v in vals])
                    block["gates"][g][key] = {
                        "cal_cov_mean": float(cov.mean()),
                        "cal_cov_sd": float(cov.std(ddof=1)),
                        "cal_err_mean": float(np.mean(
                            [v["cal_err"] for v in vals
                             if v["cal_err"] is not None] or [np.nan])),
                        "cal_misses": int(sum(not v["cal_met"] for v in vals)),
                        "plug_cov_mean": float(np.mean(
                            [v["plug_cov"] for v in vals])),
                        "plug_misses": int(sum(not v["plug_met"]
                                               for v in vals)),
                        "front_cov_mean": float(np.mean(
                            [v["front_cov"] for v in vals])),
                        "n_seeds": len(vals),
                    }
                    for kind in ("blk", "dep"):
                        if f"{kind}_cov" not in vals[0]:
                            continue
                        bcov = np.array([v[f"{kind}_cov"] for v in vals])
                        block["gates"][g][key].update({
                            f"{kind}_cov_mean": float(bcov.mean()),
                            f"{kind}_cov_sd": float(bcov.std(ddof=1)),
                            f"{kind}_err_mean": float(np.mean(
                                [v[f"{kind}_err"] for v in vals
                                 if v[f"{kind}_err"] is not None] or [np.nan])),
                            f"{kind}_misses": int(sum(not v[f"{kind}_met"]
                                                      for v in vals))})
                        if f"{kind}_conn_met" in vals[0]:
                            block["gates"][g][key][f"{kind}_conn_misses"] = \
                                int(sum(not v[f"{kind}_conn_met"]
                                        for v in vals))
            for a, b in pairs:
                if a not in gates or b not in gates:
                    continue
                name = f"{a}-{b}"
                block["contrasts"][name] = {}
                for eps in budgets:
                    key = str(float(eps))
                    block["contrasts"][name][key] = {}
                    kinds = [k for k in ("cal_cov", "blk_cov", "plug_cov",
                                         "front_cov", "dep_cov")
                             if k in rows[0]["gates"][amap][a][key]]
                    for kind in kinds:
                        d = 100 * np.array(
                            [r["gates"][amap][a][key][kind]
                             - r["gates"][amap][b][key][kind] for r in rows])
                        boot = d[rng.randint(0, len(d), (n_boot, len(d)))]
                        p = (float(wilcoxon(d).pvalue)
                             if np.any(d != 0) else 1.0)
                        lo, hi = np.percentile(boot.mean(axis=1), [2.5, 97.5])
                        block["contrasts"][name][key][kind] = {
                            "mean_points": float(d.mean()),
                            "ci95": [float(lo), float(hi)],
                            "wins": int((d > 0).sum()),
                            "ties": int((d == 0).sum()),
                            "n": int(len(d)),
                            "wilcoxon_p": p,
                        }
            for tkey in ("tiered", "tiered_blk", "tiered_dep"):
                if tkey in rows[0] and amap in rows[0][tkey]:
                    block[tkey] = {
                        g: {"coverage_mean": float(np.mean(
                            [r[tkey][amap][g]["coverage"] for r in rows])),
                            "misses_by_action": {
                                str(a): int(sum(
                                    not r[tkey][amap][g]["per_action"]
                                    .get(a, {"met": True})["met"] for r in rows))
                                for a in TIERED_BUDGETS},
                            "conn_misses_by_action": {
                                str(a): int(sum(
                                    not r[tkey][amap][g]["per_action"]
                                    .get(a, {}).get("conn_met", True)
                                    for r in rows))
                                for a in TIERED_BUDGETS},
                            **_per_action_summary(
                                [r[tkey][amap][g]["per_action"] for r in rows])}
                        for g in rows[0][tkey][amap]}
            if "aggregation_mass" in rows[0]:
                block["aggregation_mass_mean"] = float(np.mean(
                    [r["aggregation_mass"][amap] for r in rows]))
            if "confusion" in rows[0]:
                block["within_action_share_mean"] = float(np.mean(
                    [r["confusion"][amap]["within_action_share"] for r in rows]))
                block["leading_pairs_first_seed"] = \
                    rows[0]["confusion"][amap]["leading_pairs"]
            summary[split][amap] = block
    return summary


def benign_refresh_study(args):
    """
    Fast-path drops on benign captures the corpus never read, and whether
    adding one capture's flows to the training flows removes the drops on
    the other: two-action map, action gate, dependence rule.
    """
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    extra = {}
    for path in args.benign_captures:
        s, e, _, _ = P.load_corpus(path, seq_len=DECISION_PACKETS)
        extra[Path(path).stem] = switch_raw(s, e)
    amap, eps = ACTION_MAPS["2-action"], args.deploy_false_drop
    per_seed = []
    for seed in range(args.study_first_seed,
                      args.study_first_seed + args.study_seeds):
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        raw = switch_raw(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, "flow-isolated", seed)
        order = np.argsort(va, kind="stable")
        clusters = cluster_weights(fid[va])
        t_va, t_te = amap[y[va]], amap[y[te]]
        row = {"seed": seed}
        for added in [None] + list(extra):
            r_tr, y_tr = raw[tr], y[tr]
            if added:
                r_tr = np.vstack([r_tr, extra[added]])
                y_tr = np.concatenate([y_tr, np.full(len(extra[added]),
                                                     BENIGN_CLASS)])
            edges = fit_edges(r_tr)
            F_tr, F = apply_edges(r_tr, edges), apply_edges(raw, edges)
            c_tr, c_va, c_te = (cell_index(F_tr), cell_index(F[va]),
                                cell_index(F[te]))
            weights = {c: len(y_tr) / (args.num_classes * n) for c, n in
                       zip(*np.unique(y_tr, return_counts=True))}
            leaf, leaf_conf, _ = empirical_table(F_tr, y_tr, args.num_classes,
                                                 0, weights)
            counts = np.zeros((KEY_SPACE, args.num_classes))
            np.add.at(counts, (c_tr, y_tr), 1.0)
            score, act = cell_gates(counts, leaf, leaf_conf, amap)["action"]
            tau, _ = calibrate_threshold(score[c_va], act[c_va], t_va, eps,
                                         args.delta, STUDY_GRID, order,
                                         clusters)
            sel = (score[c_te] >= tau) if tau is not None else \
                np.zeros(len(te), dtype=bool)
            dropped, app_cells, by_major = {}, {}, {}
            majority = counts.argmax(axis=1)
            for name, r in extra.items():
                c = cell_index(apply_edges(r, edges))
                hit = (score[c] >= tau) if tau is not None else \
                    np.zeros(len(c), dtype=bool)
                dropped[name] = float(hit.mean())
                # Share of the dropped flows whose cell's training flows are
                # mostly reconnaissance, web, spoofing or brute force.
                app_cells[name] = float(np.isin(
                    majority[c[hit]], list(APPLICATION_ATTACKS)).mean()) \
                    if hit.any() else None
                by_major[name] = np.bincount(
                    majority[c[hit]], minlength=args.num_classes).tolist()
            row[f"with_{added}" if added else "corpus"] = {
                "coverage": float(sel.mean()),
                "selective_error": float((t_te[sel] == 0).mean())
                if sel.any() else None,
                "corpus_benign_dropped": float(sel[t_te == 0].mean()),
                "unseen_benign_dropped": dropped,
                "dropped_in_application_attack_cells": app_cells,
                "dropped_by_cell_majority": by_major}
            if added:
                continue
            # The same table under the richer maps with tiered budgets: only
            # certified drop cells drop, the milder actions only slow or
            # inspect.
            cex = {n: cell_index(apply_edges(r, edges))
                   for n, r in extra.items()}
            row["corpus"]["tiered"] = {}
            for m, am in TIERED_MAPS.items():
                s_m, a_m = cell_gates(counts, leaf, leaf_conf, am)["action"]
                drop = {n: np.zeros(len(c), dtype=bool)
                        for n, c in cex.items()}
                stop = {n: np.zeros(len(c), dtype=bool)
                        for n, c in cex.items()}
                act_te = np.zeros(len(te), dtype=bool)
                drop_te = np.zeros(len(te), dtype=bool)
                wrong_te = np.zeros(len(te), dtype=bool)
                for a, budget in TIERED_BUDGETS.items():
                    if not (a_m[c_va] == a).any():
                        continue
                    tau_a, _ = calibrate_threshold(
                        np.where(a_m[c_va] == a, s_m[c_va], -1.0), a_m[c_va],
                        am[y[va]], budget, args.delta, STUDY_GRID, order,
                        clusters)
                    if tau_a is None:
                        continue
                    for n, c in cex.items():
                        hit = (a_m[c] == a) & (s_m[c] >= tau_a)
                        stop[n] |= hit
                        if a == 1:
                            drop[n] |= hit
                    hit = (a_m[c_te] == a) & (s_m[c_te] >= tau_a)
                    act_te |= hit
                    wrong_te |= hit & (am[y[te]] != a)
                    if a == 1:
                        drop_te |= hit
                row["corpus"]["tiered"][m] = {
                    n: {"dropped": float(drop[n].mean()),
                        "any_action": float(stop[n].mean())} for n in cex}
                row["corpus"]["tiered"][m]["corpus"] = {
                    "coverage": float(act_te.mean()),
                    "drop_coverage": float(drop_te.mean()),
                    "selective_error": float(wrong_te[act_te].mean())
                    if act_te.any() else None,
                    "benign_dropped": float(drop_te[y[te] == 0].mean())}
        per_seed.append(row)
        logger.info("benign refresh seed %d: %s", seed, {
            k: {n: round(100 * v, 1) for n, v in
                row[k]["unseen_benign_dropped"].items()}
            for k in row if k != "seed"})

    def paired(a, b):
        d = 100 * (np.array(b) - np.array(a))
        boot = d[np.random.RandomState(0).randint(0, len(d), (2000, len(d)))]
        lo, hi = np.percentile(boot.mean(axis=1), [2.5, 97.5])
        return {"before": float(np.mean(a)), "after": float(np.mean(b)),
                "mean_points": float(d.mean()), "ci95": [float(lo), float(hi)],
                "lower_in": int((d < 0).sum()), "n": len(d)}
    summary = {"corpus": {
        "coverage": float(np.mean([r["corpus"]["coverage"] for r in per_seed])),
        "corpus_benign_dropped": float(np.mean(
            [r["corpus"]["corpus_benign_dropped"] for r in per_seed])),
        "unseen_benign_dropped": {n: float(np.mean(
            [r["corpus"]["unseen_benign_dropped"][n] for r in per_seed]))
            for n in extra},
        "dropped_in_application_attack_cells": {n: float(np.mean(
            [v for r in per_seed if (v := r["corpus"][
                "dropped_in_application_attack_cells"][n]) is not None]))
            for n in extra},
        "dropped_by_cell_majority": {n: np.sum(
            [r["corpus"]["dropped_by_cell_majority"][n] for r in per_seed],
            axis=0).tolist() for n in extra},
        "tiered": {m: {
            **{n: {k: float(np.mean(
                [r["corpus"]["tiered"][m][n][k] for r in per_seed]))
                for k in ("dropped", "any_action")} for n in extra},
            "corpus": {k: float(np.mean(
                [v for r in per_seed
                 if (v := r["corpus"]["tiered"][m]["corpus"][k]) is not None]))
                for k in ("coverage", "drop_coverage", "selective_error",
                          "benign_dropped")},
            "vs_two_action": {
                "coverage": paired(
                    [r["corpus"]["coverage"] for r in per_seed],
                    [r["corpus"]["tiered"][m]["corpus"]["coverage"]
                     for r in per_seed]),
                **{f"dropped_{n}": paired(
                    [r["corpus"]["unseen_benign_dropped"][n]
                        for r in per_seed],
                    [r["corpus"]["tiered"][m][n]["dropped"] for r in per_seed])
                   for n in extra}}}
            for m in TIERED_MAPS}}}
    for added in extra:
        k = f"with_{added}"
        summary[k] = {"coverage": paired(
            [r["corpus"]["coverage"] for r in per_seed],
            [r[k]["coverage"] for r in per_seed])}
        for n in extra:
            if n != added:
                summary[k][f"dropped_{n}"] = paired(
                    [r["corpus"]["unseen_benign_dropped"][n]
                        for r in per_seed],
                    [r[k]["unseen_benign_dropped"][n] for r in per_seed])
    out = {"config": {"seeds": [r["seed"] for r in per_seed],
                      "captures": {n: int(len(r)) for n, r in extra.items()},
                      "target": eps, "split": "flow-isolated",
                      "action_map": "2-action", "gate": "action"},
           "summary": summary, "per_seed": per_seed}
    Path(args.benign_refresh_out).write_text(json.dumps(out, indent=1))
    logger.info("benign refresh: %s", json.dumps(summary))


def calibration_walks(args, eps=0.01, split="flow-isolated"):
    """
    Where the dependence walk stops for the label and action gates of the
    two-action map, with every bound at the first failing threshold, and the
    validation flows and errors each gate admits at every threshold.
    """
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    amap = ACTION_MAPS["2-action"]
    grid = sorted(STUDY_GRID, reverse=True)
    start = walk_start(args.delta)
    per_seed = []
    for seed in range(args.study_first_seed,
                      args.study_first_seed + args.study_seeds):
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        raw = switch_raw(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, split, seed)
        order = np.argsort(va, kind="stable")
        units = P._group_keys(split, y, fid, X_e)[va]
        clusters = cluster_weights(units)
        F = apply_edges(raw, fit_edges(raw[tr]))
        c_tr, c_va = cell_index(F[tr]), cell_index(F[va])
        weights = {c: len(tr) / (args.num_classes * n) for c, n in
                   zip(*np.unique(y[tr], return_counts=True))}
        leaf, leaf_conf, _ = empirical_table(F[tr], y[tr], args.num_classes,
                                             0, weights)
        counts = np.zeros((KEY_SPACE, args.num_classes))
        np.add.at(counts, (c_tr, y[tr]), 1.0)
        row = {"seed": seed}
        for g, (score, act) in cell_gates(counts, leaf, leaf_conf,
                                          amap).items():
            if g not in ("label", "action"):
                continue
            s, wrong = score[c_va], act[c_va] != amap[y[va]]
            grp = cluster_bootstrap_curve(s, wrong, grid, clusters,
                                          args.delta)
            walk, last, fail = [], None, None
            for i, tau in enumerate(grid):
                sel = s >= tau
                n, k = int(sel.sum()), int(wrong[sel].sum())
                walk.append([float(tau), n, k])
                if n < start or fail is not None:
                    continue
                rec = {"tau": float(tau), "n": n, "k": k,
                       "exact": clopper_pearson_upper(k, n, args.delta),
                       "block": block_bootstrap_upper(
                           wrong[order][sel[order]], args.delta),
                       "cluster": float(grp[i]),
                       "error_connections": int(len(np.unique(
                           units[sel & wrong])))}
                if max(rec["exact"], rec["block"], rec["cluster"]) <= eps:
                    last = rec
                else:
                    fail = rec
            row[g] = {"last": last, "fail": fail, "walk": walk}
        per_seed.append(row)
        logger.info("walks seed %d: label stops at %s, action at %s", seed,
                    row["label"]["fail"] and row["label"]["fail"]["n"],
                    row["action"]["fail"] and row["action"]["fail"]["n"])

    lab = [r["label"]["fail"] for r in per_seed if r["label"]["fail"]]
    same = []
    for r in per_seed:
        f = r["label"]["fail"]
        if f:
            _, n1, k1 = min(r["action"]["walk"],
                            key=lambda w: abs(w[1] - f["n"]))
            same.append((f["k"], k1))
    summary = {
        "target": eps, "split": split, "seeds": len(per_seed),
        "label_stopped_by_block_only": sum(
            f["block"] > eps >= max(f["exact"], f["cluster"]) for f in lab),
        "label_stop_flows": [min(f["n"] for f in lab),
                             max(f["n"] for f in lab)],
        "label_stop_errors": [min(f["k"] for f in lab),
                              max(f["k"] for f in lab)],
        "label_errors_in_distinct_connections": all(
            f["error_connections"] == f["k"] for f in lab),
        "errors_at_label_stop_count": {
            "label_median": float(np.median([a for a, _ in same])),
            "action_median": float(np.median([b for _, b in same]))}}
    out = {"config": {"delta": args.delta, "walk_start_flows": start,
                      "key_levels": KEY_RADIX, "action_map": "2-action"},
           "summary": summary, "per_seed": per_seed}
    Path(args.calibration_walks_out).write_text(json.dumps(out))
    logger.info("calibration walks: %s", json.dumps(summary))


def policy_study(args):
    """
    Label against action admission over many seeds and partitioning rules,
    with baselines and richer action maps.
    """
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    scen = P.scenario_ids(y_all)
    maps = {m: {**ACTION_MAPS, **GROUPED_MAPS}[m] for m in args.study_maps}
    per_seed, curves = [], {}
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    for split in args.study_splits:
        for seed in seeds:
            keep = P.stratified_subsample(
                y_all, args.subsample_per_class, seed)
            y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
            raw = switch_raw(X_seq[keep], X_e)
            tr, va, te = P.split_indices(y, fid, X_e, split, seed)
            # Validation flows in extraction order, for block calibration,
            # and the partition's own units, for the cluster bound.
            order = np.argsort(va, kind="stable")
            units = P._group_keys(split, y, fid, X_e) \
                if split != "blocked" else None
            clusters = cluster_weights(
                units[va]) if units is not None else None
            F = apply_edges(raw, fit_edges(raw[tr]))
            c_tr, c_va, c_te = (cell_index(F[i]) for i in (tr, va, te))
            weights = {c: len(tr) / (args.num_classes * n) for c, n in
                       zip(*np.unique(y[tr], return_counts=True))}
            leaf, leaf_conf, _ = empirical_table(F[tr], y[tr],
                                                 args.num_classes, 0, weights)
            counts = np.zeros((KEY_SPACE, args.num_classes))
            np.add.at(counts, (c_tr, y[tr]), 1.0)

            row = {"split": split, "seed": seed, "n_test": int(len(te)),
                   "n_val": int(len(va)),
                   "gates": {}, "tiered": {}, "tiered_blk": {},
                   "tiered_dep": {},
                   "aggregation_mass": {m: aggregation_mass(counts, a)
                                        for m, a in maps.items()},
                   "confusion": {m: confusion_structure(
                       counts, a, [ID_TO_GROUP_LABEL.get(i, str(i))
                                   for i in range(args.num_classes)])
                                 for m, a in maps.items()}}
            if units is not None:
                # The largest unit decides much of a group partition: record
                # its size and where it fell.
                u, cnt = np.unique(units, return_counts=True)
                big = units == u[np.argmax(cnt)]
                row["largest_unit"] = {
                    "flows": int(big.sum()),
                    "classes": np.bincount(y[big], minlength=args.num_classes)
                    .tolist(),
                    "partition": ("train" if big[tr].any() else
                                  "validation" if big[va].any() else "test")}
            # The longest benign connection: all its records share a 5-tuple
            # and move together under every rule.
            ben = np.flatnonzero(y == BENIGN_CLASS)
            u, cnt = np.unique(fid[ben], return_counts=True)
            key = u[np.argmax(cnt)]
            idx = np.flatnonzero(fid == key)
            row["largest_benign_connection"] = {
                "records": int(len(idx)),
                "corpus_records": int((fid_all == key).sum()),
                "validation": int(np.isin(idx, va).sum()),
                "test": int(np.isin(idx, te).sum())}
            if scen is not None:
                # Capture scenarios that training never saw.
                sid = scen[0][keep]
                seen = np.isin(sid, np.unique(sid[tr]))
                att = y != BENIGN_CLASS
                row["unseen_scenarios"] = {
                    "validation_share": float((~seen[va]).mean()),
                    "test_share": float((~seen[te]).mean()),
                    "test_attack_share": float((~seen[te])[att[te]].mean()),
                    "test_scenarios": sorted(
                        {scen[1][i] for i in np.unique(sid[te][~seen[te]])})}
            for mname, amap in maps.items():
                t_va, t_te = amap[y[va]], amap[y[te]]
                row["gates"][mname] = {}
                for g, (score, act) in cell_gates(counts, leaf, leaf_conf,
                                                  amap).items():
                    res, curve = gate_outcomes(
                        score[c_va], act[c_va], t_va, score[c_te], act[c_te],
                        t_te, args.false_drop_budgets, args.delta, order=order,
                        groups=clusters, conn_te=fid[te])
                    row["gates"][mname][g] = {
                        str(k): v for k, v in res.items()}
                    if args.calibration_candidates and g in ("label", "action"):
                        row.setdefault("candidates", {}).setdefault(
                            mname, {})[g] = candidate_outcomes(
                            score[c_va], act[c_va], t_va, score[c_te],
                            act[c_te], t_te, args.false_drop_budgets,
                            args.delta, STUDY_GRID, order, clusters)
                    if seed == seeds[0]:
                        curves.setdefault(split, {}).setdefault(
                            mname, {})[g] = curve
                    if mname != "2-action" and g in ("label", "action",
                                                     "binary"):
                        cells = (score[c_va], act[c_va], t_va, score[c_te],
                                 act[c_te], t_te, args.delta)
                        row["tiered"].setdefault(mname, {})[g] = \
                            tiered_outcomes(*cells)
                        row["tiered_blk"].setdefault(mname, {})[g] = \
                            tiered_outcomes(*cells, order, rule="block")
                        row["tiered_dep"].setdefault(mname, {})[g] = \
                            tiered_outcomes(*cells, order, clusters,
                                            conn_te=fid[te],
                                            cells=(score, act))
            per_seed.append(row)
            m0 = next(iter(maps))
            g2 = row["gates"][m0]
            logger.info("study %s seed %d %s: 1%% coverage label %.1f%% "
                        "action %.1f%% (dependence rule: %.1f%% vs %.1f%%)",
                        split, seed, m0, 100 * g2["label"]["0.01"]["cal_cov"],
                        100 * g2["action"]["0.01"]["cal_cov"],
                        100 * g2["label"]["0.01"]["dep_cov"],
                        100 * g2["action"]["0.01"]["dep_cov"])

    out = {
        "config": {"seeds": list(seeds), "splits": list(args.study_splits),
                   "budgets": list(args.false_drop_budgets),
                   "delta": args.delta, "decision_packets": DECISION_PACKETS,
                   "key_levels": KEY_RADIX,
                   "block_bootstrap": {"blocks": BOOT_BLOCKS,
                                       "resamples": BOOT_RESAMPLES,
                                       "order": "extraction order of the "
                                                "validation flows"},
                   "dependence_rule": {
                       "bound": "max(exact binomial, block bootstrap, "
                                "cluster bootstrap over partition units)",
                       "units": "flow-isolated: 5-tuple; context-disjoint: "
                                "signature component; blocked: none",
                       "walk_start_flows": walk_start(args.delta),
                       "chosen_on": "development seeds 0-29"},
                   "subsample_per_class": args.subsample_per_class,
                   "action_maps": {k: v.tolist() for k, v in maps.items()},
                   "tiered_budgets": TIERED_BUDGETS},
        "summary": summarise_study(per_seed, args.false_drop_budgets),
        "curves_first_seed": curves,
        "per_seed": per_seed,
    }
    Path(args.study_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.study_out).write_text(json.dumps(out))
    logger.info("Policy study written to %s", args.study_out)


def _quantile_grid(score, levels=401):
    f = score[score > -1e8]
    return (np.unique(np.quantile(f, np.linspace(0, 1, levels)))
            if len(f) else np.array([1.0]))


def _count_gates(counts, weights, amap):
    """cell_gates for any partition given its training class counts."""
    w = np.array([weights.get(c, 1.0) for c in range(counts.shape[1])])
    leaf = (counts * w[None, :]).argmax(axis=1)
    support = counts.sum(axis=1)
    leaf_conf = wilson_lower(counts[np.arange(len(counts)), leaf], support)
    return cell_gates(counts, leaf, leaf_conf, amap)


def _gbdt_scores(Z_tr, y_tr, Z_evals, seed, weights):
    """Fine-label and binary gradient boosting; gates read from raw margins."""
    from lightgbm import LGBMClassifier
    fine = LGBMClassifier(n_estimators=300, num_leaves=31, class_weight=weights,
                          random_state=seed, verbose=-1, n_jobs=8).fit(Z_tr, y_tr)
    binary = LGBMClassifier(n_estimators=300, num_leaves=31, random_state=seed,
                            verbose=-1, n_jobs=8).fit(Z_tr, (y_tr != 0).astype(int))
    out = []
    for Z in Z_evals:
        s = fine.predict(Z, raw_score=True)
        logp = s - np.logaddexp.reduce(s, axis=1, keepdims=True)
        top = logp.argmax(axis=1)
        b = binary.predict(Z, raw_score=True)
        out.append({
            "gbdt_label": np.where(top != 0, logp[np.arange(len(s)), top], -1e9),
            "gbdt_action": np.where(-logp[:, 0] >= np.log(2), -logp[:, 0], -1e9),
            "gbdt_binary": np.where(b >= 0, b, -1e9),
            "p_benign": 1.0 / (1.0 + np.exp(b)),
        })
    return out


def _mtl_scores(Z_tr, y_tr, Z_evals, seed, weights, epochs=30):
    """Multi-task MLP: a fine-label head and an action head on one trunk."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    L_tr = np.log1p(Z_tr)
    mu, sd = L_tr.mean(0), L_tr.std(0) + 1e-9

    def tens(Z):
        return torch.tensor((np.log1p(Z) - mu) / sd, dtype=torch.float32)
    X, y = tens(Z_tr), torch.tensor(y_tr, dtype=torch.long)
    a = (y != 0).float()
    n_cls = int(y_tr.max()) + 1
    trunk = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 128), torch.nn.ReLU())
    head_c, head_a = torch.nn.Linear(128, n_cls), torch.nn.Linear(128, 1)
    params = [*trunk.parameters(), *head_c.parameters(), *head_a.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(
        [weights.get(c, 1.0) for c in range(n_cls)], dtype=torch.float32))
    bce = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            h = trunk(X[b])
            loss = ce(head_c(h), y[b]) + bce(head_a(h).squeeze(1), a[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    out = []
    with torch.no_grad():
        for Z in Z_evals:
            h = trunk(tens(Z))
            lc = torch.log_softmax(head_c(h), 1).numpy()
            la = head_a(h).squeeze(1).numpy()
            top = lc.argmax(1)
            out.append({
                "mtl_label": np.where(top != 0, lc[np.arange(len(lc)), top], -1e9),
                "mtl_action": np.where(la >= 0, la, -1e9)})
    return out


BASELINE_SWEEP_LEVELS = (6, 8, 12)


def baselines_study(args):
    """
    The action gate against deployable and controller-grade alternatives on
    the same five registers, for the two-action policy: a direct binary
    classifier, a multi-task network with fine and action heads, cost-
    sensitive plug-in rules, label gating on each model, and range-match
    decision trees a switch can hold. Also the key-resolution sweep.
    """
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    amap = ACTION_MAPS["2-action"]
    levels = KEY_RADIX
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    per_seed = []
    for seed in seeds:
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        raw = switch_raw(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, "flow-isolated", seed)
        order = np.argsort(va, kind="stable")
        clusters = cluster_weights(fid[va])
        t_va, t_te = amap[y[va]], amap[y[te]]
        weights = {c: len(tr) / (args.num_classes * n) for c, n in
                   zip(*np.unique(y[tr], return_counts=True))}
        scores = {}           # name -> (validation score, test score, grid)

        for L in (levels,) + BASELINE_SWEEP_LEVELS:
            set_key_levels(L)
            F = apply_edges(raw, fit_edges(raw[tr]))
            c_tr, c_va, c_te = (cell_index(F[i]) for i in (tr, va, te))
            counts = np.zeros((KEY_SPACE, args.num_classes))
            np.add.at(counts, (c_tr, y[tr]), 1.0)
            g = _count_gates(counts, weights, amap)
            tag = "table" if L == levels else f"table{L}"
            for name in ("label", "action"):
                s = g[name][0]
                scores[f"{tag}_{name}"] = (s[c_va], s[c_te], STUDY_GRID)
            if L == levels:
                sup = counts.sum(axis=1)
                ben_freq = np.where(
                    sup > 0, counts[:, 0] / np.maximum(sup, 1), 1.0)
                table_pb = ben_freq[c_te]
        set_key_levels(levels)

        # Decision trees a switch can hold as range-match entries.
        for name, target in (("tree_binary", y[tr] != 0), ("tree_fine", y[tr])):
            tree = DecisionTreeClassifier(max_leaf_nodes=256, min_samples_leaf=30,
                                          random_state=seed).fit(raw[tr], target)
            lf = [tree.apply(raw[i]) for i in (tr, va, te)]
            counts = np.zeros((tree.tree_.node_count, args.num_classes))
            np.add.at(counts, (lf[0], y[tr]), 1.0)
            g = _count_gates(counts, weights, amap)
            if name == "tree_binary":
                s = g["binary"][0]
                scores["tree_binary"] = (s[lf[1]], s[lf[2]], STUDY_GRID)
            else:
                for gname in ("label", "action"):
                    s = g[gname][0]
                    scores[f"tree_{gname}"] = (s[lf[1]], s[lf[2]], STUDY_GRID)

        gb_va, gb_te = _gbdt_scores(raw[tr], y[tr], (raw[va], raw[te]), seed,
                                    weights)
        for k in ("gbdt_label", "gbdt_action", "gbdt_binary"):
            scores[k] = (gb_va[k], gb_te[k], _quantile_grid(gb_va[k]))
        mt_va, mt_te = _mtl_scores(raw[tr], y[tr], (raw[va], raw[te]), seed,
                                   weights)
        for k in ("mtl_label", "mtl_action"):
            scores[k] = (mt_va[k], mt_te[k], _quantile_grid(mt_va[k]))

        row = {"seed": seed, "methods": {}}
        ones_va, ones_te = np.ones(len(va), int), np.ones(len(te), int)
        for k, (s_va, s_te, grid) in scores.items():
            res, _ = gate_outcomes(s_va, ones_va, t_va, s_te, ones_te, t_te,
                                   args.false_drop_budgets, args.delta, grid,
                                   order, clusters, conn_te=fid[te])
            row["methods"][k] = {str(e): v for e, v in res.items()}
        # Cost-sensitive plug-in rules: drop when the estimated benign
        # probability is at most the budget, with no calibration step.
        for k, pb in (("table_costsens", table_pb),
                      ("gbdt_costsens", gb_te["p_benign"])):
            row["methods"][k] = {}
            for eps in args.false_drop_budgets:
                sel = pb <= eps
                err = float((t_te[sel] == 0).mean()) if sel.any() else None
                row["methods"][k][str(float(eps))] = {
                    "plug_cov": float(sel.mean()), "plug_err": err,
                    "plug_met": err is None or err <= eps}
        per_seed.append(row)
        m = row["methods"]
        logger.info("baselines seed %d at 1%%: table action %.1f%% (dependence "
                    "%.1f%%), label %.1f%%, tree binary %.1f%%, gbdt binary "
                    "%.1f%%, gbdt action %.1f%%, mtl action %.1f%%", seed,
                    *(100 * m[k]["0.01"][f] for k, f in (
                        ("table_action", "cal_cov"), ("table_action", "dep_cov"),
                        ("table_label", "cal_cov"), ("tree_binary", "cal_cov"),
                        ("gbdt_binary", "cal_cov"), ("gbdt_action", "cal_cov"),
                        ("mtl_action", "cal_cov"))))

    from scipy.stats import wilcoxon
    rng = np.random.RandomState(0)
    summary = {}
    for k in per_seed[0]["methods"]:
        summary[k] = {}
        for eps in args.false_drop_budgets:
            key = str(float(eps))
            vals = [r["methods"][k][key] for r in per_seed]
            entry = {}
            for kind in ("cal", "blk", "plug", "front", "dep"):
                if f"{kind}_cov" not in vals[0]:
                    continue
                cov = np.array([v[f"{kind}_cov"] for v in vals])
                entry[f"{kind}_cov_mean"] = float(cov.mean())
                entry[f"{kind}_cov_sd"] = float(cov.std(ddof=1))
                if kind != "front":
                    entry[f"{kind}_misses"] = int(sum(not v[f"{kind}_met"]
                                                      for v in vals))
                if k != "table_action" and f"{kind}_cov" in \
                        per_seed[0]["methods"]["table_action"][key]:
                    d = 100 * np.array(
                        [r["methods"]["table_action"][key][f"{kind}_cov"]
                         - r["methods"][k][key][f"{kind}_cov"] for r in per_seed])
                    boot = d[rng.randint(
                        0, len(d), (2000, len(d)))].mean(axis=1)
                    entry[f"table_action_minus_{kind}"] = {
                        "mean_points": float(d.mean()),
                        "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                        "wins": int((d > 0).sum()), "n": int(len(d)),
                        "wilcoxon_p": float(wilcoxon(d).pvalue)
                        if np.any(d != 0) else 1.0}
            summary[k][key] = entry
    out = {"config": {"seeds": list(seeds), "split": "flow-isolated",
                      "policy": "2-action", "key_levels": levels,
                      "sweep_levels": list(BASELINE_SWEEP_LEVELS),
                      "budgets": list(args.false_drop_budgets),
                      "delta": args.delta,
                      "features": "the five switch registers at packet 10",
                      "trees": "256 leaves, min 30 rows per leaf, range-match "
                               "deployable",
                      "gbdt": "LightGBM, 300 trees, 31 leaves",
                      "mtl": "MLP 128-128, fine-label and action heads, 30 epochs",
                      "block_bootstrap": {"blocks": BOOT_BLOCKS,
                                          "resamples": BOOT_RESAMPLES},
                      "dependence_rule": {"units": "5-tuple",
                                          "walk_start_flows":
                                              walk_start(args.delta)}},
           "summary": summary, "per_seed": per_seed}
    Path(args.baselines_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.baselines_out).write_text(json.dumps(out))
    logger.info("Baselines study written to %s", args.baselines_out)


DIRECT_ACTION_MAPS = ("3-action", "4-action", "flood-inspect")


def _action_gate_from_counts(counts_a):
    """Score and action per cell from counts over actions: Wilson lower
    bound of the leading action's share, ineligible where that is allow."""
    support = counts_a.sum(axis=1)
    best = counts_a.argmax(axis=1)
    conf = wilson_lower(counts_a.max(axis=1), support)
    return np.where(best != ACTION_ALLOW, conf, -1.0), best


def _action_gate_from_proba(p_fine, amap, n_actions):
    """Pool class probabilities into action probabilities; score is the
    leading action's probability, ineligible where that is allow."""
    p_act = np.zeros((len(p_fine), n_actions))
    for c, a in enumerate(amap):
        p_act[:, a] += p_fine[:, c]
    best = p_act.argmax(axis=1)
    return np.where(best != ACTION_ALLOW, p_act.max(axis=1), -1.0), best


def direct_action_study(args):
    """
    Pooling a fine-label model by action against a model trained directly
    on the action labels, for the multi-action maps, with the same features,
    partitions and per-action calibration. For the cell table the two are
    the same arithmetic, which the study checks; for trees and gradient
    boosting they are different fits.
    """
    import time
    from lightgbm import LGBMClassifier
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    maps = {m: {**ACTION_MAPS, **GROUPED_MAPS}[m] for m in DIRECT_ACTION_MAPS}
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    per_seed = []
    for seed in seeds:
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        raw = switch_raw(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, "flow-isolated", seed)
        order = np.argsort(va, kind="stable")
        clusters = cluster_weights(fid[va])
        weights = {c: len(tr) / (args.num_classes * n) for c, n in
                   zip(*np.unique(y[tr], return_counts=True))}
        F = apply_edges(raw, fit_edges(raw[tr]))
        c_tr, c_va, c_te = (cell_index(F[i]) for i in (tr, va, te))
        counts = np.zeros((KEY_SPACE, args.num_classes))
        np.add.at(counts, (c_tr, y[tr]), 1.0)
        # Fine-label models are fitted once and reused for every map.
        t0 = time.perf_counter()
        gb_fine = LGBMClassifier(n_estimators=300, num_leaves=31,
                                 class_weight=weights, random_state=seed,
                                 verbose=-1, n_jobs=8).fit(raw[tr], y[tr])
        fit_fine = time.perf_counter() - t0
        p_fine = {k: gb_fine.predict_proba(raw[i]) for k, i in
                  (("va", va), ("te", te))}
        tree_fine = DecisionTreeClassifier(
            max_leaf_nodes=256, min_samples_leaf=30,
            random_state=seed).fit(raw[tr], y[tr])
        lf_fine = {k: tree_fine.apply(raw[i]) for k, i in
                   (("tr", tr), ("va", va), ("te", te))}
        tree_counts = np.zeros((tree_fine.tree_.node_count, args.num_classes))
        np.add.at(tree_counts, (lf_fine["tr"], y[tr]), 1.0)
        row = {"seed": seed, "fit_seconds": {"gbdt_fine": fit_fine},
               "maps": {}}
        for mname, amap in maps.items():
            n_act = int(amap.max()) + 1
            a_tr, t_va, t_te = amap[y[tr]], amap[y[va]], amap[y[te]]
            methods = {}
            # Cell table: pooled fine counts against direct action counts.
            t0 = time.perf_counter()
            pooled = _count_gates(counts, weights, amap)["action"]
            compile_pooled = time.perf_counter() - t0
            counts_a = np.zeros((KEY_SPACE, n_act))
            np.add.at(counts_a, (c_tr, a_tr), 1.0)
            direct = _action_gate_from_counts(counts_a)
            methods["table_pooled"] = (pooled[0][c_va], pooled[1][c_va],
                                       pooled[0][c_te], pooled[1][c_te],
                                       pooled)
            methods["table_direct"] = (direct[0][c_va], direct[1][c_va],
                                       direct[0][c_te], direct[1][c_te],
                                       direct)
            table_diff = float(np.abs(pooled[0] - direct[0]).max())
            # Range-match tree: fine leaves pooled, or a tree on actions.
            g = _count_gates(tree_counts, weights, amap)["action"]
            methods["tree_pooled"] = (g[0][lf_fine["va"]], g[1][lf_fine["va"]],
                                      g[0][lf_fine["te"]], g[1][lf_fine["te"]],
                                      None)
            tree_a = DecisionTreeClassifier(
                max_leaf_nodes=256, min_samples_leaf=30,
                random_state=seed).fit(raw[tr], a_tr)
            lf = [tree_a.apply(raw[i]) for i in (tr, va, te)]
            ca = np.zeros((tree_a.tree_.node_count, n_act))
            np.add.at(ca, (lf[0], a_tr), 1.0)
            gd = _action_gate_from_counts(ca)
            methods["tree_direct"] = (gd[0][lf[1]], gd[1][lf[1]],
                                      gd[0][lf[2]], gd[1][lf[2]], None)
            # Gradient boosting: pooled class probabilities, or a direct fit.
            gp = {k: _action_gate_from_proba(p_fine[k], amap, n_act)
                  for k in ("va", "te")}
            methods["gbdt_pooled"] = (gp["va"][0], gp["va"][1],
                                      gp["te"][0], gp["te"][1], None)
            t0 = time.perf_counter()
            gb_a = LGBMClassifier(
                n_estimators=300, num_leaves=31, random_state=seed,
                class_weight={a: len(tr) / (n_act * n) for a, n in
                              zip(*np.unique(a_tr, return_counts=True))},
                verbose=-1, n_jobs=8).fit(raw[tr], a_tr)
            fit_direct = time.perf_counter() - t0
            ident = np.eye(n_act, dtype=int)[gb_a.classes_]
            gdp = {}
            for k, i in (("va", va), ("te", te)):
                p = gb_a.predict_proba(raw[i]) @ ident
                gdp[k] = _action_gate_from_proba(p, np.arange(n_act), n_act)
            methods["gbdt_direct"] = (gdp["va"][0], gdp["va"][1],
                                      gdp["te"][0], gdp["te"][1], None)
            out = {"table_max_abs_score_diff": table_diff,
                   "compile_seconds_pooled_table": compile_pooled,
                   "fit_seconds_gbdt_direct": fit_direct, "methods": {}}
            for name, (s_va, a_va, s_te, a_te, cells) in methods.items():
                res, _ = gate_outcomes(s_va, a_va, t_va, s_te, a_te, t_te,
                                       args.false_drop_budgets, args.delta,
                                       STUDY_GRID, order, clusters,
                                       conn_te=fid[te])
                out["methods"][name] = {
                    "single": {str(e): v for e, v in res.items()},
                    "tiered_dep": tiered_outcomes(
                        s_va, a_va, t_va, s_te, a_te, t_te, args.delta,
                        order, clusters, conn_te=fid[te], cells=cells)}
            row["maps"][mname] = out
        per_seed.append(row)
        logger.info("direct action seed %d: %s", seed, "; ".join(
            f"{m}: pooled {100 * r['methods']['gbdt_pooled']['tiered_dep']['coverage']:.1f}% "
            f"direct {100 * r['methods']['gbdt_direct']['tiered_dep']['coverage']:.1f}%"
            for m, r in row["maps"].items()))

    rng = np.random.RandomState(0)
    summary = {}
    for mname in maps:
        summary[mname] = {"table_max_abs_score_diff": float(max(
            r["maps"][mname]["table_max_abs_score_diff"] for r in per_seed))}
        names = per_seed[0]["maps"][mname]["methods"]
        for name in names:
            t = [r["maps"][mname]["methods"][name]["tiered_dep"]
                 for r in per_seed]
            entry = {"tiered_coverage_mean": float(np.mean(
                [x["coverage"] for x in t])),
                "tiered_misses_by_action": {
                    str(a): int(sum(not x["per_action"].get(a, {"met": True})
                                    ["met"] for x in t))
                    for a in TIERED_BUDGETS},
                **_per_action_summary([x["per_action"] for x in t])}
            for eps in args.false_drop_budgets:
                k = str(float(eps))
                vals = [r["maps"][mname]["methods"][name]["single"][k]
                        for r in per_seed]
                entry[f"dep_cov_mean_{k}"] = float(np.mean(
                    [v["dep_cov"] for v in vals]))
                entry[f"dep_misses_{k}"] = int(sum(not v["dep_met"]
                                                   for v in vals))
                entry[f"front_cov_mean_{k}"] = float(np.mean(
                    [v["front_cov"] for v in vals]))
            summary[mname][name] = entry
        for fam in ("tree", "gbdt"):
            d = 100 * np.array([
                r["maps"][mname]["methods"][f"{fam}_pooled"]["tiered_dep"]["coverage"]
                - r["maps"][mname]["methods"][f"{fam}_direct"]["tiered_dep"]["coverage"]
                for r in per_seed])
            boot = d[rng.randint(0, len(d), (2000, len(d)))].mean(axis=1)
            summary[mname][f"{fam}_pooled_minus_direct_tiered"] = {
                "mean_points": float(d.mean()),
                "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                "wins": int((d > 0).sum()), "n": int(len(d))}
    out = {"config": {"seeds": list(seeds), "split": "flow-isolated",
                      "maps": {k: v.tolist() for k, v in maps.items()},
                      "tiered_budgets": TIERED_BUDGETS,
                      "budgets": list(args.false_drop_budgets),
                      "delta": args.delta, "key_levels": KEY_RADIX,
                      "gbdt": "LightGBM, 300 trees, 31 leaves",
                      "trees": "256 leaves, min 30 rows per leaf"},
           "summary": summary, "per_seed": per_seed}
    Path(args.direct_action_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.direct_action_out).write_text(json.dumps(out))
    logger.info("Direct action study written to %s", args.direct_action_out)


CONTROLLER_FPR = (0.01, 0.02, 0.05)


def fit_report_controller(raw_tr, attack_tr, seed):
    """Action-level adjudicator over registers the escalation report carries."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=8,
                          random_state=seed, verbose=-1).fit(raw_tr, attack_tr)


def fpr_threshold(score, benign, already, beta):
    """
    Lowest threshold among the observed scores whose benign block rate,
    counting flows already blocked, is <= beta; None if even the highest
    score exceeds the budget.
    """
    room = int(np.floor(beta * benign.sum() + 1e-9)) - \
        int((already & benign).sum())
    if room < 0:
        return None
    grid = np.unique(score)
    cand = np.sort(score[benign & ~already])[::-1]
    if room >= len(cand):
        return float(grid[0])
    above = grid[grid > cand[room]]
    return float(above[0]) if len(above) else None


TWO_TIER_OFFLOADS = (0.1, 0.2, 0.3, 0.4)
REPORT_SETS = {"report5": 5, "report12": 12}


def fpr_threshold_block(score, benign, already, beta, order, delta=0.05,
                        n_candidates=200, clusters=None):
    """
    Like fpr_threshold, but a threshold is accepted only while an upper bound
    on the benign block rate stays within beta. The bound follows the
    dependence rule of threshold_bounds: the largest of the exact binomial
    bound, a block bootstrap quantile over contiguous benign validation flows
    in extraction order, and, with ``clusters`` (cluster_weights of the
    validation flows), a cluster bootstrap quantile over the partition's
    units. Candidates are quantiles of the scores not already blocked,
    walked from the most conservative down, stopping at the first failure.
    """
    ben_order = order[benign[order]]
    free = score[~already & np.isfinite(score)]
    if len(ben_order) == 0 or len(free) == 0:
        return None
    cands = np.unique(np.quantile(free, np.linspace(0, 1, n_candidates)))[::-1]
    blocks = np.array_split(np.arange(len(ben_order)),
                            min(BOOT_BLOCKS, len(ben_order)))
    n = np.array([len(b) for b in blocks], dtype=float)
    pick = np.random.RandomState(0).randint(0, len(blocks),
                                            (BOOT_RESAMPLES, len(blocks)))
    if clusters is not None:
        gidx, W = clusters
        n_g = np.bincount(gidx[benign], minlength=W.shape[1]).astype(float)
        nb = W @ n_g
    n_ben = int(benign.sum())
    best = None
    for tau in cands:
        blocked = already | (score >= tau)
        hit = blocked[ben_order]
        k = np.array([hit[b].sum() for b in blocks], dtype=float)
        upper = max(np.quantile(k[pick].sum(1) / n[pick].sum(1), 1 - delta),
                    clopper_pearson_upper(int(hit.sum()), n_ben, delta))
        if clusters is not None:
            k_g = np.bincount(gidx[benign & blocked], minlength=W.shape[1])
            kb = W @ k_g.astype(float)
            upper = max(upper, np.quantile(
                np.where(nb > 0, kb / np.maximum(nb, 1.0), 0.0), 1 - delta))
        if upper > beta:
            break
        best = float(tau)
    return best


def _split_curve(sw_va, sw_te, s_va, s_te, att_va, att_te, beta):
    """
    Walk the switch threshold down from 'off'. At each setting the controller
    threshold fills what is left of the benign budget, once chosen on
    validation flows and once read on test flows (the matched comparison).
    Stops when the switch alone exceeds the budget on validation.
    """
    out = []
    zv, zt = np.zeros(len(sw_va), bool), np.zeros(len(sw_te), bool)
    for tau in [np.inf] + list(STUDY_GRID[::-1]):
        fv, ft = sw_va >= tau, sw_te >= tau
        if fv[~att_va].mean() > beta:
            break
        tc = fpr_threshold(np.where(fv, -np.inf, s_va), ~att_va, fv, beta)
        bv = fv | ((s_va >= tc) if tc is not None else zv)
        bt = ft | ((s_te >= tc) if tc is not None else zt)
        matched = None
        if ft[~att_te].mean() <= beta:
            tm = fpr_threshold(np.where(ft, -np.inf, s_te), ~att_te, ft, beta)
            matched = float((ft | ((s_te >= tm) if tm is not None else zt))
                            [att_te].mean())
        out.append({"switch_threshold": float(tau),
                    "offload_val": float(fv.mean()), "offload": float(ft.mean()),
                    "recall_val": float(bv[att_va].mean()),
                    "recall": float(bt[att_te].mean()),
                    "benign": float(bt[~att_te].mean()),
                    "recall_matched": matched})
    return out


def two_tier_study(args):
    """
    Switch admission followed by controller adjudication of the deferred
    flows, where the controller sees only the emitted report (the five
    policy registers, or all twelve registers of the extended report).
    Compared with the same controller adjudicating every flow, under one
    total benign block budget. The switch threshold is chosen jointly with
    the controller threshold: the largest validation offload whose
    validation recall stays within one point of the controller alone.
    """
    from scipy.stats import wilcoxon
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    amap = ACTION_MAPS["2-action"]
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    rows = []
    for split in args.two_tier_splits:
        for seed in seeds:
            keep = P.stratified_subsample(
                y_all, args.subsample_per_class, seed)
            y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
            regs = report_registers(X_seq[keep], X_e)
            raw = regs[:, :5]
            tr, va, te = P.split_indices(y, fid, X_e, split, seed)
            order = np.argsort(va, kind="stable")
            units = P._group_keys(split, y, fid, X_e) \
                if split != "blocked" else None
            clusters = cluster_weights(
                units[va]) if units is not None else None
            F = apply_edges(raw, fit_edges(raw[tr]))
            k_tr, k_va, k_te = (cell_index(F[i]) for i in (tr, va, te))
            weights = {c: len(tr) / (args.num_classes * n) for c, n in
                       zip(*np.unique(y[tr], return_counts=True))}
            leaf, leaf_conf, _ = empirical_table(F[tr], y[tr],
                                                 args.num_classes, 0, weights)
            counts = np.zeros((KEY_SPACE, args.num_classes))
            np.add.at(counts, (k_tr, y[tr]), 1.0)
            score, act = cell_gates(counts, leaf, leaf_conf, amap)["action"]
            att_va, att_te = y[va] != 0, y[te] != 0
            tau_s, _ = calibrate_threshold(score[k_va], act[k_va],
                                           att_va.astype(int),
                                           args.deploy_false_drop, args.delta,
                                           STUDY_GRID, order, clusters)
            sw_va, sw_te = score[k_va], score[k_te]
            dep_te = sw_te >= tau_s if tau_s is not None else np.zeros(
                len(te), bool)
            row = {"split": split, "seed": seed,
                   "deployed_switch_threshold": tau_s,
                   "deployed_offload": float(dep_te.mean()),
                   "deployed_switch_benign_rate": float(dep_te[~att_te].mean()),
                   "deployed_switch_attack_rate": float(dep_te[att_te].mean()),
                   "largest_benign_5tuple_test": int(np.unique(
                       fid[te][~att_te], return_counts=True)[1].max()),
                   "controllers": {}}
            for cname, width in REPORT_SETS.items():
                model = fit_report_controller(regs[tr][:, :width],
                                              (y[tr] != 0).astype(int), seed)
                s_va = model.predict(regs[va][:, :width], raw_score=True)
                s_te = model.predict(regs[te][:, :width], raw_score=True)
                row["controllers"][cname] = {}
                for beta in CONTROLLER_FPR:
                    curve = _split_curve(sw_va, sw_te, s_va, s_te, att_va,
                                         att_te, beta)
                    base = curve[0]
                    ok = [p for p in curve
                          if p["recall_val"] >= base["recall_val"] - 0.01]
                    joint = max(ok, key=lambda p: p["offload_val"])
                    at = {}
                    for o in TWO_TIER_OFFLOADS:
                        cand = [p for p in curve if p["offload"] >= o
                                and p["recall_matched"] is not None]
                        at[str(o)] = (min(cand, key=lambda p: p["offload"])
                                      ["recall_matched"] if cand else None)
                    feas = [p for p in curve if p["recall_matched"] is not None]
                    best = max(feas, key=lambda p: (
                        p["recall_matched"], p["offload"]))
                    # Deployable arms with the dependence-aware benign bound.
                    zv = np.zeros(len(va), bool)
                    blk = {}
                    for arm, tau in (("alone_blk", np.inf),
                                     ("joint_blk", joint["switch_threshold"])):
                        fv, ft = sw_va >= tau, sw_te >= tau
                        tc = fpr_threshold_block(np.where(fv, -np.inf, s_va),
                                                 ~att_va, fv | zv, beta, order,
                                                 args.delta, clusters=clusters)
                        bt = ft | ((s_te >= tc) if tc is not None
                                   else np.zeros(len(te), bool))
                        # The same rate over connections: a benign 5-tuple
                        # counts once, blocked if any of its records is.
                        ben_fid = fid[te][~att_te]
                        blk[arm] = {"recall": float(bt[att_te].mean()),
                                    "benign": float(bt[~att_te].mean()),
                                    "benign_connections": float(
                                        len(np.unique(ben_fid[bt[~att_te]]))
                                        / len(np.unique(ben_fid))),
                                    "offload": float(ft.mean())}
                    row["controllers"][cname][str(beta)] = {
                        "alone": base, "joint": joint, **blk,
                        "matched_at_offload": at,
                        "matched_best": {"recall": best["recall_matched"],
                                         "offload": best["offload"]},
                        "max_feasible_offload": float(curve[-1]["offload"])}
            rows.append(row)
            c = row["controllers"]["report12"]["0.02"]
            logger.info("two-tier %s seed %d at 2%% benign: alone %.1f%%, joint "
                        "%.1f%% with %.1f%% offload (report12); matched alone "
                        "%.1f%% vs best split %.1f%% at %.1f%% offload", split,
                        seed, 100 * c["alone"]["recall"], 100 *
                        c["joint"]["recall"],
                        100 * c["joint"]["offload"],
                        100 * c["alone"]["recall_matched"],
                        100 * c["matched_best"]["recall"],
                        100 * c["matched_best"]["offload"])

    rng = np.random.RandomState(0)

    def paired(d):
        d = 100 * np.asarray(d, dtype=float)
        boot = d[rng.randint(0, len(d), (2000, len(d)))].mean(axis=1)
        return {"mean_points": float(d.mean()),
                "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                "n": int(len(d)),
                "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}

    def mean(v):
        v = [x for x in v if x is not None]
        return {"mean": float(np.mean(v)) if v else None, "n": len(v)}

    summary = {}
    for split in args.two_tier_splits:
        sub = [r for r in rows if r["split"] == split]
        summary[split] = {
            "deployed_offload": mean([r["deployed_offload"] for r in sub]),
            "deployed_switch_benign_rate": mean(
                [r["deployed_switch_benign_rate"] for r in sub])}
        for cname in REPORT_SETS:
            summary[split][cname] = {}
            for beta in CONTROLLER_FPR:
                b = str(beta)
                pts = [r["controllers"][cname][b] for r in sub]
                s = {"alone_recall": mean([p["alone"]["recall"] for p in pts]),
                     "alone_benign": mean([p["alone"]["benign"] for p in pts]),
                     "alone_benign_over_budget": int(sum(
                         p["alone"]["benign"] > beta for p in pts)),
                     "joint_recall": mean([p["joint"]["recall"] for p in pts]),
                     "joint_benign": mean([p["joint"]["benign"] for p in pts]),
                     "joint_benign_over_budget": int(sum(
                         p["joint"]["benign"] > beta for p in pts)),
                     "joint_offload": mean([p["joint"]["offload"] for p in pts]),
                     "joint_minus_alone": paired(
                         [p["joint"]["recall"] - p["alone"]["recall"] for p in pts]),
                     "alone_blk_recall": mean([p["alone_blk"]["recall"] for p in pts]),
                     "alone_blk_benign": mean([p["alone_blk"]["benign"] for p in pts]),
                     "alone_blk_over_budget": int(sum(
                         p["alone_blk"]["benign"] > beta for p in pts)),
                     "alone_blk_benign_connections": mean(
                         [p["alone_blk"]["benign_connections"] for p in pts]),
                     "alone_blk_connections_over_budget": int(sum(
                         p["alone_blk"]["benign_connections"] > beta
                         for p in pts)),
                     "joint_blk_recall": mean([p["joint_blk"]["recall"] for p in pts]),
                     "joint_blk_benign": mean([p["joint_blk"]["benign"] for p in pts]),
                     "joint_blk_over_budget": int(sum(
                         p["joint_blk"]["benign"] > beta for p in pts)),
                     "joint_blk_benign_connections": mean(
                         [p["joint_blk"]["benign_connections"] for p in pts]),
                     "joint_blk_connections_over_budget": int(sum(
                         p["joint_blk"]["benign_connections"] > beta
                         for p in pts)),
                     "joint_blk_offload": mean([p["joint_blk"]["offload"] for p in pts]),
                     "joint_blk_minus_alone_blk": paired(
                         [p["joint_blk"]["recall"] - p["alone_blk"]["recall"]
                          for p in pts]),
                     "matched_alone_recall": mean(
                         [p["alone"]["recall_matched"] for p in pts]),
                     "matched_best_recall": mean(
                         [p["matched_best"]["recall"] for p in pts]),
                     "matched_best_offload": mean(
                         [p["matched_best"]["offload"] for p in pts]),
                     "matched_best_minus_alone": paired(
                         [p["matched_best"]["recall"] - p["alone"]["recall_matched"]
                          for p in pts]),
                     "max_feasible_offload": mean(
                         [p["max_feasible_offload"] for p in pts]),
                     "matched_at_offload": {}}
                for o in TWO_TIER_OFFLOADS:
                    pairs = [(p["matched_at_offload"][str(o)],
                              p["alone"]["recall_matched"]) for p in pts
                             if p["matched_at_offload"][str(o)] is not None]
                    s["matched_at_offload"][str(o)] = {
                        "recall": mean([a for a, _ in pairs]),
                        "minus_alone": paired([a - c for a, c in pairs])
                        if len(pairs) > 1 else None}
                summary[split][cname][b] = s
        # Does the extended report help? Same split, same seeds.
        summary[split]["report12_minus_report5"] = {
            str(beta): {
                "alone": paired([r["controllers"]["report12"][str(beta)]["alone"]["recall_matched"]
                                 - r["controllers"]["report5"][str(beta)]["alone"]["recall_matched"]
                                 for r in sub]),
                "joint": paired([r["controllers"]["report12"][str(beta)]["joint"]["recall"]
                                 - r["controllers"]["report5"][str(beta)]["joint"]["recall"]
                                 for r in sub])}
            for beta in CONTROLLER_FPR}
    Path(args.two_tier_out).write_text(json.dumps(
        {"config": {"seeds": list(seeds), "splits": list(args.two_tier_splits),
                    "switch_budget": args.deploy_false_drop,
                    "switch_calibration": "block",
                    "benign_block_budgets": list(CONTROLLER_FPR),
                    "report_sets": {"report5": list(FEATURE_NAMES),
                                    "report12": list(FEATURE_NAMES)
                                    + list(EXTRA_REPORT_FEATURES)},
                    "joint_rule": "largest validation offload whose validation "
                                  "recall is within 1 point of the controller "
                                  "alone",
                    "controller_model": "binary LightGBM, 200 trees, report "
                                        "registers at packet 10"},
         "summary": summary, "per_seed": rows}))
    logger.info("Two-tier study written to %s", args.two_tier_out)


def export_report_controller(raw, y, tr, va, te, fast_va, beta, seed, out_dir,
                             order=None, clusters=None):
    """
    Write the adjudicator the controller runs on escalation reports. With
    ``order`` its benign budget is calibrated with the dependence bound of
    fpr_threshold_block; without it, with the plug-in validation rate.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from controller.report_adjudicator import REPORT_FEATURES, ReportAdjudicator
    model = fit_report_controller(raw[tr], (y[tr] != 0).astype(int), seed)
    s_va = model.predict(raw[va], raw_score=True)
    if order is None:
        tau = fpr_threshold(np.where(fast_va, -np.inf, s_va), y[va] == 0,
                            fast_va, beta)
    else:
        tau = fpr_threshold_block(np.where(fast_va, -np.inf, s_va), y[va] == 0,
                                  fast_va, beta, order, clusters=clusters)
    path = out_dir / "controller_model.json"
    path.write_text(json.dumps({
        "features": list(REPORT_FEATURES),
        "threshold": None if tau is None else float(tau),
        "benign_block_budget": beta, "seed": seed,
        "model": model.booster_.dump_model()}))
    adj = ReportAdjudicator(path)
    sample = te[:500]
    ref = model.predict(raw[sample], raw_score=True)
    mine = np.array([adj.score(dict(zip(REPORT_FEATURES, r)))
                     for r in raw[sample]])
    err = float(np.abs(ref - mine).max())
    logger.info("Report adjudicator: threshold %s at %.0f%% benign budget, "
                "plain-Python scorer max |diff| %.2e", tau, 100 * beta, err)
    return {"threshold": None if tau is None else float(tau),
            "benign_block_budget": beta, "scorer_max_abs_diff": err,
            "calibration": "plug-in" if order is None else "dependence"}


CONTROLLER_GRID = {"trees": (50, 100, 200, 400), "leaves": (15, 31, 63)}


def controller_sensitivity(args):
    """
    The report adjudicator over tree counts and leaf limits: attack recall
    at a validation-selected and at a matched benign block rate, the size of
    the exported tree dump, and the scoring time per report. The plain-Python
    scorer of the controller is timed on the deployed configuration.
    """
    import time
    from lightgbm import LGBMClassifier
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from controller.report_adjudicator import REPORT_FEATURES, ReportAdjudicator
    X_seq, X_ent, y_all, fid_all = P.load_corpus(args.data,
                                                 seq_len=DECISION_PACKETS)
    seeds = range(args.study_first_seed,
                  args.study_first_seed + args.study_seeds)
    per_seed = []
    for seed in seeds:
        keep = P.stratified_subsample(y_all, args.subsample_per_class, seed)
        y, X_e, fid = y_all[keep], X_ent[keep], fid_all[keep]
        regs = report_registers(X_seq[keep], X_e)
        tr, va, te = P.split_indices(y, fid, X_e, "flow-isolated", seed)
        att_va, att_te = y[va] != 0, y[te] != 0
        none_va = np.zeros(len(va), dtype=bool)
        none_te = np.zeros(len(te), dtype=bool)
        row = {"seed": seed, "configs": {}}
        for n_est in CONTROLLER_GRID["trees"]:
            for leaves in CONTROLLER_GRID["leaves"]:
                t0 = time.perf_counter()
                model = LGBMClassifier(n_estimators=n_est, num_leaves=leaves,
                                       n_jobs=8, random_state=seed,
                                       verbose=-1).fit(regs[tr],
                                                       (y[tr] != 0).astype(int))
                fit_s = time.perf_counter() - t0
                s_va = model.predict(regs[va], raw_score=True)
                t0 = time.perf_counter()
                s_te = model.predict(regs[te], raw_score=True)
                batch_us = 1e6 * (time.perf_counter() - t0) / len(te)
                sample = regs[te][:200]
                t0 = time.perf_counter()
                for r in sample:
                    model.predict(r[None, :], raw_score=True)
                single_us = 1e6 * (time.perf_counter() - t0) / len(sample)
                entry = {"fit_seconds": fit_s,
                         "model_bytes": len(model.booster_.model_to_string()),
                         "leaves_total": int(sum(
                             t["num_leaves"] for t in
                             model.booster_.dump_model()["tree_info"])),
                         "batch_us_per_report": batch_us,
                         "single_us_per_report": single_us, "budgets": {}}
                if n_est == 200 and leaves == 31:
                    # The deployed configuration, scored as the controller
                    # scores it: a tree dump read by plain Python.
                    path = Path(args.controller_sensitivity_out).parent / \
                        f"_controller_probe_s{seed}.json"
                    path.write_text(json.dumps({
                        "features": list(REPORT_FEATURES), "threshold": 0.0,
                        "benign_block_budget": 0.0, "seed": seed,
                        "model": model.booster_.dump_model()}))
                    adj = ReportAdjudicator(path)
                    rows_ = [dict(zip(REPORT_FEATURES, r)) for r in sample]
                    t0 = time.perf_counter()
                    for r in rows_:
                        adj.score(r)
                    entry["python_scorer_us_per_report"] = \
                        1e6 * (time.perf_counter() - t0) / len(rows_)
                    entry["dump_bytes"] = path.stat().st_size
                    path.unlink()
                for beta in CONTROLLER_FPR:
                    thr = fpr_threshold(s_va, ~att_va, none_va, beta)
                    blk = s_te >= thr if thr is not None else none_te
                    thr_m = fpr_threshold(s_te, ~att_te, none_te, beta)
                    blk_m = s_te >= thr_m if thr_m is not None else none_te
                    entry["budgets"][str(beta)] = {
                        "recall": float(blk[att_te].mean()),
                        "benign_rate": float(blk[~att_te].mean()),
                        "matched_recall": float(blk_m[att_te].mean())}
                row["configs"][f"{n_est}x{leaves}"] = entry
        per_seed.append(row)
        c = row["configs"]
        logger.info("controller sensitivity seed %d: matched recall at 1%% "
                    "50x31 %.1f%%, 200x31 %.1f%%, 400x63 %.1f%%; 200x31 %d B, "
                    "%.0f us/report batched", seed,
                    100 * c["50x31"]["budgets"]["0.01"]["matched_recall"],
                    100 * c["200x31"]["budgets"]["0.01"]["matched_recall"],
                    100 * c["400x63"]["budgets"]["0.01"]["matched_recall"],
                    c["200x31"]["model_bytes"],
                    c["200x31"]["batch_us_per_report"])

    rng = np.random.RandomState(0)
    summary = {}
    ref = "200x31"
    for cfg in per_seed[0]["configs"]:
        rows = [r["configs"][cfg] for r in per_seed]
        entry = {k: float(np.mean([x[k] for x in rows])) for k in
                 ("fit_seconds", "model_bytes", "leaves_total",
                  "batch_us_per_report", "single_us_per_report")}
        if "python_scorer_us_per_report" in rows[0]:
            entry["python_scorer_us_per_report"] = float(np.mean(
                [x["python_scorer_us_per_report"] for x in rows]))
            entry["dump_bytes"] = float(np.mean([x["dump_bytes"] for x in rows]))
        for beta in CONTROLLER_FPR:
            b = str(beta)
            for k in ("recall", "benign_rate", "matched_recall"):
                entry[f"{k}_mean_{b}"] = float(np.mean(
                    [x["budgets"][b][k] for x in rows]))
            d = 100 * np.array([
                r["configs"][cfg]["budgets"][b]["matched_recall"]
                - r["configs"][ref]["budgets"][b]["matched_recall"]
                for r in per_seed])
            boot = d[rng.randint(0, len(d), (2000, len(d)))].mean(axis=1)
            entry[f"matched_recall_minus_ref_{b}"] = {
                "mean_points": float(d.mean()),
                "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}
        summary[cfg] = entry
    out = {"config": {"seeds": list(seeds), "split": "flow-isolated",
                      "reference": ref, "grid": CONTROLLER_GRID,
                      "budgets": list(CONTROLLER_FPR),
                      "features": "the twelve report registers at packet 10",
                      "timing": "LightGBM predict on one CPU container; the "
                                "plain-Python scorer on 200 single reports"},
           "summary": summary, "per_seed": per_seed}
    Path(args.controller_sensitivity_out).parent.mkdir(parents=True,
                                                       exist_ok=True)
    Path(args.controller_sensitivity_out).write_text(json.dumps(out))
    logger.info("Controller sensitivity written to %s",
                args.controller_sensitivity_out)


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
    ap.add_argument("--deploy-calibration", default="dependence",
                    choices=("dependence", "block", "binomial"),
                    help="bound used to select the deployed threshold")
    ap.add_argument("--deploy-map", default="2-action",
                    choices=["2-action", *TIERED_MAPS],
                    help="also export a tiered table for this action map")
    ap.add_argument("--tiered-out-dir", default="p4/generated/tiered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--out-dir", default="p4/generated")
    ap.add_argument("--results", default="results/dataplane_classifier.json")
    ap.add_argument("--from-bmv2", default=None, nargs="+",
                    help="Evaluate the features the switch measured, from one or "
                         "more collections produced by replay_attack.py --collect")
    ap.add_argument("--measured-seeds", type=int, default=1,
                    help="Flow-split seeds per collection in --from-bmv2 mode")
    ap.add_argument("--study", action="store_true",
                    help="Compare admission rules over many seeds, splits, "
                         "baselines and action maps instead of compiling")
    ap.add_argument("--study-seeds", type=int, default=30)
    ap.add_argument("--study-first-seed", type=int, default=30,
                    help="Seeds 0 to 29 were used for development; reported "
                         "studies start at 30")
    ap.add_argument("--key-levels", type=int, default=KEY_RADIX,
                    help="Quantisation levels per table key field")
    ap.add_argument("--baselines", action="store_true",
                    help="Compare the action gate with binary, multi-task, "
                         "cost-sensitive and tree baselines")
    ap.add_argument("--baselines-out", default="results/baselines_study.json")
    ap.add_argument("--study-splits", nargs="+", default=list(STUDY_SPLITS),
                    choices=list(P.SPLIT_MODES))
    ap.add_argument("--study-out", default="results/policy_study.json")
    ap.add_argument("--study-maps", nargs="+", default=list(ACTION_MAPS),
                    choices=[*ACTION_MAPS, *GROUPED_MAPS])
    ap.add_argument("--calibration-candidates", action="store_true",
                    help="Policy study: also record every candidate "
                         "calibration rule (for development seeds)")
    ap.add_argument("--two-tier", action="store_true",
                    help="Evaluate switch admission plus a controller that "
                         "sees only the emitted report")
    ap.add_argument("--two-tier-out", default="results/two_tier_study.json")
    ap.add_argument("--two-tier-splits", nargs="+",
                    default=["flow-isolated", "context-disjoint"],
                    choices=list(P.SPLIT_MODES))
    ap.add_argument("--controller-fpr", type=float, default=0.02,
                    help="Benign block budget for the exported report adjudicator")
    ap.add_argument("--map-study", action="store_true",
                    help="Gain of the action gate over random action maps")
    ap.add_argument("--map-study-out", default="results/map_study.json")
    ap.add_argument("--map-count", type=int, default=30,
                    help="Random maps per number of attack actions")
    ap.add_argument("--map-target", type=float, default=0.01)
    ap.add_argument("--benign-refresh", action="store_true",
                    help="Drops on unseen benign captures, with and without "
                         "adding one of them to the training flows")
    ap.add_argument("--benign-captures", nargs="+",
                    default=["data/sequences/benign_unseen2.h5",
                             "data/sequences/benign_unseen3.h5"])
    ap.add_argument("--benign-refresh-out",
                    default="results/benign_refresh.json")
    ap.add_argument("--calibration-walks", action="store_true",
                    help="Where the dependence walk stops for the label and "
                         "action gates at a 1%% target, per seed")
    ap.add_argument("--calibration-walks-out",
                    default="results/calibration_walks.json")
    ap.add_argument("--direct-action", action="store_true",
                    help="Pooled fine-label models against models trained "
                         "directly on the action labels, multi-action maps")
    ap.add_argument("--direct-action-out",
                    default="results/direct_action_study.json")
    ap.add_argument("--controller-sensitivity", action="store_true",
                    help="Report adjudicator over tree counts and leaf limits")
    ap.add_argument("--controller-sensitivity-out",
                    default="results/controller_sensitivity.json")
    args = ap.parse_args()
    set_key_levels(args.key_levels)

    if args.direct_action:
        direct_action_study(args)
        return
    if args.controller_sensitivity:
        controller_sensitivity(args)
        return
    if args.baselines:
        baselines_study(args)
        return
    if args.study:
        policy_study(args)
        return
    if args.two_tier:
        two_tier_study(args)
        return
    if args.map_study:
        map_study(args)
        return
    if args.benign_refresh:
        benign_refresh_study(args)
        return
    if args.calibration_walks:
        calibration_walks(args)
        return

    if args.from_bmv2:
        if len(args.from_bmv2) == 1 and args.measured_seeds == 1:
            out = evaluate_measured(Path(args.from_bmv2[0]), args.seed,
                                    args.num_classes)
        else:
            out = aggregate_measured(args.from_bmv2, args.measured_seeds,
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
    policy_entry_conf = np.where(cell_action == ACTION_DROP,
                                 cell_action_conf, 0.0)
    # The same fixed-sequence walk with a dependence-aware bound: the rule
    # chosen on development seeds, or the earlier block rule.
    order_va = np.argsort(va, kind="stable")
    units = P._group_keys(args.split, y, flow_ids, X_ent) \
        if args.split != "blocked" else None
    clusters = cluster_weights(units[va]) if units is not None else None
    eligible = np.where(cell_action == ACTION_DROP, cell_action_conf, -1.0)
    rule = "block" if args.deploy_calibration == "block" else "dependence"
    tau_blk, ub_blk = calibrate_threshold(
        eligible[code_va], cell_action[code_va],
        (y[va] != BENIGN_CLASS).astype(np.int64), args.deploy_false_drop,
        args.delta, grid, order_va, clusters, rule)
    if tau_blk is None:
        raise RuntimeError("dependence-aware calibration failed at the "
                           "deployed budget")
    sel_blk = eligible[code_te] >= tau_blk
    block_point = {
        "rule": rule,
        "threshold": float(tau_blk), "calibration_upper_bound": ub_blk,
        "coverage": float(sel_blk.mean()),
        "false_drop_rate": float((y[te][sel_blk] == BENIGN_CLASS).mean())
        if sel_blk.any() else None}
    if args.deploy_calibration in ("block", "dependence"):
        deploy_policy, deploy_tau = block_point, float(tau_blk)
    else:
        deploy_policy = deploy["action_confidence"]
        deploy_tau = deploy_policy["threshold"]
    logger.info("deployed threshold %.2f (%s calibration): coverage %.1f%%, "
                "selective drop error %.2f%%", deploy_tau,
                args.deploy_calibration, 100 * deploy_policy["coverage"],
                100 * (deploy_policy["false_drop_rate"] or 0.0))
    summary = export_entries(policy_cls, policy_entry_conf, deploy_tau,
                             args.num_classes, Path(args.out_dir))
    summary.update({
        "policy": "drop only; allow and uncertain cells escalate",
        "false_drop_budget": args.deploy_false_drop,
        "calibrated_threshold": deploy_tau,
        "calibration": args.deploy_calibration,
    })
    export_bucketise(edges, Path(args.out_dir))
    logger.info("Exported %d entries of a %d-cell key space",
                summary["entries_installed"], summary["key_space"])
    raw12 = report_registers(X_seq, X_ent)
    # The adjudicator's benign budget uses the same dependence-aware bound
    # unless the binomial rule was requested.
    dep = args.deploy_calibration != "binomial"
    adjudicator = export_report_controller(
        raw12, y, tr, va, te, policy_entry_conf[code_va] >= deploy_tau,
        args.controller_fpr, args.seed, Path(args.out_dir),
        order_va if dep else None, clusters if dep else None)
    tiered = None
    if args.deploy_map != "2-action":
        tiered = export_tiered(F, y, tr, va, te, cell_cls, cell_conf,
                               TIERED_MAPS[args.deploy_map], edges, args,
                               Path(args.tiered_out_dir),
                               units[va] if units is not None else None)

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
            "levels": int(KEY_RADIX),
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
            "report_adjudicator": adjudicator,
            "false_drop_budget": args.deploy_false_drop,
            "calibration": args.deploy_calibration,
            "threshold": deploy_tau,
            "coverage": deploy_policy["coverage"],
            "false_drop_rate": deploy_policy["false_drop_rate"],
            "binomial_point": deploy["action_confidence"],
            "block_point": block_point,
            "entries_installed": summary["entries_installed"],
        },
        "tiered_policy": tiered,
        "table": summary,
        "confusion_matrix": confusion_matrix(
            y[te], composed, labels=range(args.num_classes)).tolist(),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", args.results)


if __name__ == "__main__":
    main()
