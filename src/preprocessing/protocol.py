"""
protocol.py - Evaluation protocol for the flow-sequence corpus.

Centralises every decision that determines what a reported number means:

  * how flows are partitioned into train / validation / test,
  * how the context (entropy) channel is presented to the model,
  * where normalisation statistics are estimated,
  * how the corpus is subsampled for controlled comparisons.

All training and baseline scripts import from here so that a single
configuration object fully determines the protocol, and so that two runs
that differ in one flag differ in exactly one variable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np

# Context-signature quantisation used by the context-disjoint protocol,
# in bits. Two flows whose six context features agree to within this
# resolution are treated as sharing a measurement window.
CONTEXT_SIGNATURE_DECIMALS = 1

ENTROPY_MODES = ("real", "none", "class-marginal", "shuffle")
SPLIT_MODES = ("flow-isolated", "context-disjoint", "blocked", "random")


# ── Corpus loading ────────────────────────────────────────────────────

def load_corpus(h5_path: str, seq_len: Optional[int] = None):
    """
    Load the flow corpus.

    Returns (X_seq, X_ent, y, flow_ids) where X_seq is (N, L, 3),
    X_ent is (N, 6) and flow_ids is an array of str.
    """
    with h5py.File(h5_path, "r") as hf:
        X_seq = hf["sequences"][:].astype(np.float32)
        X_ent = hf["entropy"][:].astype(np.float32)
        y = hf["labels"][:].astype(np.int64)
        raw = hf["flow_ids"][:]
    flow_ids = np.array(
        [f.decode() if isinstance(f, bytes) else str(f) for f in raw])

    if seq_len is not None and seq_len < X_seq.shape[1]:
        X_seq = np.ascontiguousarray(X_seq[:, :seq_len, :])
    return X_seq, X_ent, y, flow_ids


def packet_counts(X_seq: np.ndarray) -> np.ndarray:
    """Number of observed (non-padded) packets per flow."""
    return (X_seq[:, :, 0] != 0).sum(axis=1).astype(np.int64)


def scenario_ids(y: np.ndarray, log_path: str = "results/log_extraction_v3.txt"
                 ) -> Optional[Tuple[np.ndarray, list]]:
    """
    Capture scenario (CICIoT2023 directory) of every corpus flow.

    The extractor wrote the scenarios in sorted directory order and logged
    each one's flow count, so the corpus is a concatenation of scenario runs.
    Returns (scenario index per flow, scenario names), or None when the log
    is missing or its counts do not reproduce ``y`` exactly. Scenarios with
    more than 30,000 qualifying flows were sampled to 30,000 in random order
    by the extractor; the others keep the order in which flows completed.
    """
    import re
    from pathlib import Path
    p = Path(log_path)
    if not p.exists():
        return None
    done = dict((n, int(k)) for n, k in re.findall(
        r"\[([^\]]+)\] DONE: (\d+) flows", p.read_text()))
    names = sorted(done)
    if sum(done.values()) != len(y):
        return None
    sid = np.concatenate([np.full(done[n], i) for i, n in enumerate(names)])
    # Every scenario must carry a single label that matches the corpus.
    for i in range(len(names)):
        if len(np.unique(y[sid == i])) != 1:
            return None
    return sid, names


# ── Subsampling ───────────────────────────────────────────────────────

def stratified_subsample(y: np.ndarray, cap_per_class: Optional[int],
                         seed: int) -> np.ndarray:
    """
    Indices of a per-class capped subsample.

    Classes smaller than the cap are kept whole, so the operation reduces
    the dominance of the majority classes without discarding any minority
    flow. Returns sorted indices.
    """
    if cap_per_class is None:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > cap_per_class:
            idx = rng.choice(idx, cap_per_class, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


# ── Grouping keys ─────────────────────────────────────────────────────

def context_signature(X_ent: np.ndarray,
                      decimals: int = CONTEXT_SIGNATURE_DECIMALS) -> np.ndarray:
    """
    Group key identifying flows that share a context measurement window.

    The context features of a flow are computed over a window of the
    packet stream, so two flows whose windows overlap may receive similar
    context vectors. Quantising that vector defines a proxy grouping for a
    sensitivity analysis. Because packet-to-window membership was not
    retained, this signature cannot prove that all overlapping windows are
    grouped together.
    """
    q = np.round(X_ent, decimals)
    # Python's hash() of bytes is salted per process, which made the split irreproducible.
    return np.unique(q, axis=0, return_inverse=True)[1].astype(np.int64).ravel()


def _group_keys(mode: str, y: np.ndarray, flow_ids: np.ndarray,
                X_ent: np.ndarray) -> Optional[np.ndarray]:
    if mode == "random":
        return None
    if mode == "flow-isolated":
        return flow_ids
    if mode == "context-disjoint":
        # A flow is tied to its bidirectional connection *and* to every
        # other flow measured in the same context window.
        sig = context_signature(X_ent)
        return _union_groups(flow_ids, sig)
    raise ValueError(f"unknown split mode: {mode}")


def _union_groups(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Connected components of the graph linking samples that share a value
    of ``a`` or a value of ``b``. Implemented as union-find over the two
    key spaces, which is linear in the number of samples.
    """
    n = len(a)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n_a = ai.max() + 1
    parent = np.arange(n_a + bi.max() + 1)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in range(n):
        ra, rb = find(ai[i]), find(n_a + bi[i])
        if ra != rb:
            parent[rb] = ra

    return np.array([find(ai[i]) for i in range(n)], dtype=np.int64)


# ── Splitting ─────────────────────────────────────────────────────────

def split_indices(y: np.ndarray, flow_ids: np.ndarray, X_ent: np.ndarray,
                  mode: str, seed: int,
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition indices into train / val / test under the requested protocol.

    ``random`` assigns individual flows and is retained only to quantify
    the optimism it introduces. The other modes assign whole groups, so
    no group contributes to more than one partition.
    """
    rng = np.random.RandomState(seed)
    if mode == "blocked":
        return _blocked_split(y, train_ratio, val_ratio)
    groups = _group_keys(mode, y, flow_ids, X_ent)

    if groups is None:
        parts = ([], [], [])
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_tr = int(len(idx) * train_ratio)
            n_va = int(len(idx) * val_ratio)
            parts[0].extend(idx[:n_tr])
            parts[1].extend(idx[n_tr:n_tr + n_va])
            parts[2].extend(idx[n_tr + n_va:])
        return tuple(np.sort(np.array(p)) for p in parts)

    # Assign whole groups, stratifying on the group's majority label so
    # that the class mix of each partition stays close to the corpus.
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    boundaries = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    members = [order[boundaries[g]:boundaries[g + 1]]
               for g in range(len(uniq))]
    group_label = np.array([np.bincount(y[m]).argmax() for m in members])

    parts = ([], [], [])
    for c in np.unique(group_label):
        gidx = np.where(group_label == c)[0]
        rng.shuffle(gidx)
        n_tr = int(len(gidx) * train_ratio)
        n_va = int(len(gidx) * val_ratio)
        for g in gidx[:n_tr]:
            parts[0].append(members[g])
        for g in gidx[n_tr:n_tr + n_va]:
            parts[1].append(members[g])
        for g in gidx[n_tr + n_va:]:
            parts[2].append(members[g])

    return tuple(
        np.sort(np.concatenate(p)) if p else np.array([], dtype=np.int64)
        for p in parts)


def _blocked_split(y: np.ndarray, train_ratio: float, val_ratio: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extraction-order blocks: inside each contiguous label run of the corpus,
    the earliest flows train, the next validate and the latest test. Rows
    must be in corpus order, which ``stratified_subsample`` preserves.
    """
    parts = ([], [], [])
    for run in np.split(np.arange(len(y)), np.flatnonzero(np.diff(y)) + 1):
        n_tr = int(len(run) * train_ratio)
        n_va = int(len(run) * val_ratio)
        parts[0].append(run[:n_tr])
        parts[1].append(run[n_tr:n_tr + n_va])
        parts[2].append(run[n_tr + n_va:])
    return tuple(np.concatenate(p) for p in parts)


# ── Normalisation (estimated on train only) ───────────────────────────

@dataclass
class Normalizer:
    """z-score statistics estimated on the training partition alone."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    ent_mean: np.ndarray
    ent_std: np.ndarray

    @classmethod
    def fit(cls, X_seq: np.ndarray, X_ent: np.ndarray) -> "Normalizer":
        valid = X_seq[:, :, 0] != 0            # observed packet positions
        seq_mean = np.zeros(X_seq.shape[2], dtype=np.float32)
        seq_std = np.ones(X_seq.shape[2], dtype=np.float32)
        for c in range(X_seq.shape[2]):
            obs = X_seq[:, :, c][valid]
            # Leave any genuinely constant observed channel untouched.
            if obs.size and obs.std() > 1e-6:
                seq_mean[c] = obs.mean()
                seq_std[c] = obs.std()

        ent_std = X_ent.std(axis=0)
        ent_mean = np.where(ent_std > 1e-6, X_ent.mean(axis=0), 0.0)
        ent_std = np.where(ent_std > 1e-6, ent_std, 1.0)
        return cls(seq_mean, seq_std,
                   ent_mean.astype(np.float32), ent_std.astype(np.float32))

    def apply_seq(self, X: np.ndarray) -> np.ndarray:
        """Standardise observed packets and leave padding at zero."""
        valid = (X[:, :, 0] != 0)[:, :, None]
        return np.where(valid, (X - self.seq_mean) / self.seq_std, 0.0
                        ).astype(np.float32)

    def apply_ent(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.ent_mean) / self.ent_std).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {k: np.asarray(v).tolist() for k, v in asdict(self).items()}


# ── Context channel presentation ──────────────────────────────────────

def apply_entropy_mode(X_ent: np.ndarray, y: np.ndarray, mode: str,
                       seed: int, reference: Optional[np.ndarray] = None,
                       reference_y: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """
    Return the context matrix actually shown to a model.

    ``real``            the measured context vector of each flow.
    ``none``            a constant zero vector; the context channel is removed.
    ``class-marginal``  an oracle diagnostic selected using the flow's true
                        class. It preserves the label-conditioned marginal
                        while destroying per-flow correspondence, but is not
                        a deployable detector input.
    ``shuffle``         a vector drawn from the corpus-wide empirical
                        distribution, removing all association with the label.

    ``reference`` supplies the pool that substitutes are drawn from and
    must come from the training partition so that the controls do not
    themselves import information across the split.
    """
    if mode == "real":
        return X_ent
    if mode == "none":
        return np.zeros_like(X_ent)

    rng = np.random.RandomState(seed)
    pool = X_ent if reference is None else reference
    pool_y = y if reference_y is None else reference_y

    if mode == "shuffle":
        return pool[rng.randint(0, len(pool), size=len(X_ent))]

    if mode == "class-marginal":
        out = np.empty_like(X_ent)
        for c in np.unique(y):
            tgt = np.where(y == c)[0]
            src = np.where(pool_y == c)[0]
            if len(src) == 0:
                src = np.arange(len(pool))
            out[tgt] = pool[rng.choice(src, len(tgt), replace=True)]
        return out

    raise ValueError(f"unknown entropy mode: {mode}")


# ── Reporting helpers ─────────────────────────────────────────────────

def split_report(y: np.ndarray, parts: Sequence[np.ndarray],
                 groups: Optional[np.ndarray] = None) -> Dict:
    """Summary of a realised split, stored alongside every result file."""
    names = ("train", "val", "test")
    out: Dict[str, Dict] = {}
    for name, idx in zip(names, parts):
        out[name] = {
            "n": int(len(idx)),
            "class_counts": np.bincount(y[idx], minlength=8).tolist(),
        }
    if groups is not None:
        out["n_groups"] = int(len(np.unique(groups)))
        overlap = 0
        seen: Dict[int, int] = {}
        for p, idx in enumerate(parts):
            for g in np.unique(groups[idx]):
                if g in seen and seen[g] != p:
                    overlap += 1
                seen[g] = p
        out["groups_crossing_partitions"] = int(overlap)
    return out
