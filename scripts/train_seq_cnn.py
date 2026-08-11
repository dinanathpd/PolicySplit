"""
train_seq_cnn.py — Train the 1D-CNN on PCAP packet sequences.

Uses the DataPlaneCNN in sequence mode: (batch, seq_len=100, 3) input
with 1D convolutions over the temporal packet dimension.

Supports both flow-isolated splitting (default, no leakage) and
random row-level splitting (for leakage demonstration).

Usage:
    # Flow-isolated (default)
    python scripts/train_seq_cnn.py --data data/sequences/entropy_flows_v2.h5

    # Random split (leakage demo)
    python scripts/train_seq_cnn.py --data data/sequences/entropy_flows_v2.h5 \
        --split random --name seq_cnn_random

Reads from:  data/sequences/entropy_flows_v2.h5
Outputs:     models/<name>_best.pt
             results/<name>_results.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402
from src.utils.metrics import compute_metrics, print_classification_report  # noqa: E402
from src.models.mamba_controller import FocalLoss  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Deeper 1D-CNN for packet sequences
# -----------------------------------------------------------------------

class SequenceCNN(nn.Module):
    """
    1D-CNN for per-packet flow classification.

    Input: (batch, seq_len=100, 3) — [log_pkt_len, log_IAT, direction]
    Architecture:
        Conv1D(3→64, k=5) → BN → ReLU → MaxPool(2)
        Conv1D(64→128, k=3) → BN → ReLU → MaxPool(2)
        Conv1D(128→256, k=3) → BN → ReLU → GlobalAvgPool
        FC(256→128) → ReLU → Dropout(0.3) → FC(128→num_classes)
    """

    def __init__(self, num_classes: int = 8, seq_len: int = 100):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pool → (B, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (batch, seq_len, 3) → logits (batch, num_classes)"""
        x = x.transpose(1, 2)  # → (B, 3, seq_len)
        x = self.conv_block(x)
        return self.classifier(x)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_h5(path):
    with h5py.File(path, "r") as hf:
        X = hf["sequences"][:].astype(np.float32)
        y = hf["labels"][:].astype(np.int64)
        flow_ids = None
        if "flow_ids" in hf:
            raw = hf["flow_ids"][:]
            flow_ids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in raw]
    return X, y, flow_ids


def stratified_split(X, y, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Row-level stratified split (leakage risk)."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def flow_isolated_split(X, y, flow_ids, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Flow-isolated stratified split (no leakage)."""
    rng = np.random.RandomState(seed)
    fid_to_idx = {}
    for i, fid in enumerate(flow_ids):
        fid_to_idx.setdefault(fid, []).append(i)

    fid_label = {fid: y[idxs[0]] for fid, idxs in fid_to_idx.items()}
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        cls_fids = [fid for fid, lab in fid_label.items() if lab == c]
        rng.shuffle(cls_fids)
        n = len(cls_fids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for fid in cls_fids[:n_train]:
            train_idx.extend(fid_to_idx[fid])
        for fid in cls_fids[n_train:n_train + n_val]:
            val_idx.extend(fid_to_idx[fid])
        for fid in cls_fids[n_train + n_val:]:
            test_idx.extend(fid_to_idx[fid])

    train_idx, val_idx, test_idx = np.array(
        train_idx), np.array(val_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def balance_classes(X, y, max_per_class=None, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    counts = {c: (y == c).sum() for c in classes}
    min_count = min(counts.values())
    target = max_per_class if max_per_class else min_count

    balanced_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) >= target:
            chosen = rng.choice(idx, size=target, replace=False)
        else:
            chosen = rng.choice(idx, size=target, replace=True)
        balanced_idx.append(chosen)
    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            preds = model(X_b).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
    return compute_metrics(np.array(all_labels), np.array(all_preds), num_classes)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN on PCAP packet sequences")
    parser.add_argument("--data", type=str,
                        default="data/sequences/entropy_flows_v2.h5")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--split", type=str, default="flow-isolated",
                        choices=["flow-isolated", "random"])
    parser.add_argument("--name", type=str, default="seq_cnn")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N epochs (default: 50)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Load ---
    logger.info("Loading %s", args.data)
    X_all, y_all, flow_ids = load_h5(args.data)

    classes, counts = np.unique(y_all, return_counts=True)
    logger.info("Raw distribution:")
    for c, cnt in zip(classes, counts):
        logger.info("  %s (%d): %d", ID_TO_GROUP_LABEL.get(
            int(c), str(c)), c, cnt)

    # --- Balance ---
    if args.balance:
        X_all, y_all = balance_classes(
            X_all, y_all, args.max_per_class, args.seed)
        logger.info("Balanced → %d samples", len(y_all))
        flow_ids = None  # balancing breaks flow structure

    # --- Split ---
    if args.split == "flow-isolated" and flow_ids is not None:
        logger.info("Using FLOW-ISOLATED split")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = flow_isolated_split(
            X_all, y_all, flow_ids, seed=args.seed)
    else:
        logger.info("Using ROW-LEVEL split (%s)",
                    "fallback" if flow_ids is None else "by request")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(
            X_all, y_all, seed=args.seed)
    logger.info("Train: %d, Val: %d, Test: %d",
                len(y_train), len(y_val), len(y_test))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).long()),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).long()),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                      torch.from_numpy(y_test).long()),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = SequenceCNN(num_classes=args.num_classes,
                        seq_len=args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("SequenceCNN params: %d (%.2f M)", n_params, n_params / 1e6)

    # --- Loss / Optimizer ---
    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Training ---
    best_val_f1 = 0.0
    patience_ctr = 0
    start_epoch = 1
    history = {"train_loss": [], "val_f1": [], "val_acc": []}

    # --- Resume from checkpoint ---
    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        patience_ctr = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        logger.info("  Resumed at epoch %d, best_val_f1=%.4f",
                    start_epoch, best_val_f1)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_b, y_b in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                             leave=False):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_m = evaluate(model, val_loader, device, args.num_classes)
        val_f1 = val_m["macro_f1"]
        val_acc = val_m["accuracy"]
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        logger.info("Epoch %d/%d | loss=%.4f | val_acc=%.4f | val_f1=%.4f | %.1fs",
                    epoch, args.epochs, avg_loss, val_acc, val_f1, time.time() - t0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"{args.name}_best.pt"))
            logger.info("  → New best val_f1=%.4f, saved!", val_f1)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Periodic checkpoint
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"{args.name}_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "patience_counter": patience_ctr,
                "history": history,
                "args": vars(args),
            }, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path)

    # --- Test ---
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, f"{args.name}_best.pt"),
                   map_location=device, weights_only=True))
    test_m = evaluate(model, test_loader, device, args.num_classes)
    class_names = [ID_TO_GROUP_LABEL.get(i, str(i))
                   for i in range(args.num_classes)]
    logger.info("=== Test Results (%s, %s split) ===", args.name, args.split)
    print_classification_report(test_m, class_names)

    cm = test_m.get("confusion_matrix")
    results = {
        "model": args.name,
        "split": args.split,
        "best_val_f1": best_val_f1,
        "test": {
            "accuracy": test_m["accuracy"],
            "macro_f1": test_m["macro_f1"],
            "weighted_f1": test_m["weighted_f1"],
            "per_class": test_m["per_class"],
        },
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "history": history,
    }
    with open(os.path.join(args.results_dir, f"{args.name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved.")


if __name__ == "__main__":
    main()
