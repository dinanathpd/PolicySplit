"""
train_bimamba.py — Train BiMamba + Entropy on packet sequences.

This is the main training script for the novel BiMamba architecture:
  - Bidirectional Selective State Space Model (forward + backward scans)
  - Entropy-augmented features (src_ip_H, dst_ip_H, port entropies)
  - Attention pooling over temporal dimension
  - FocalLoss with class weights for imbalanced data

Usage:
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data data/sequences/entropy_flows.h5 \
        --d-model 128 --n-layers 4 --epochs 80

Reads:  data/sequences/entropy_flows.h5  (from pcap_to_entropy_sequences.py)
Output: models/bimamba_best.pt
        results/bimamba_results.json
"""

from src.models.mamba_controller import FocalLoss
from src.models.bimamba_controller import BiMambaController
from src.utils.metrics import compute_metrics, print_classification_report
from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL
from src.preprocessing import protocol as P
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch
import numpy as np
import h5py
import time
import logging
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────

class EntropyFlowDataset(Dataset):
    """Dataset that serves (sequences, entropy_features, label) tuples."""

    def __init__(self, sequences, entropy, labels, augment=False):
        self.sequences = torch.from_numpy(sequences).float()
        self.entropy = torch.from_numpy(entropy).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        ent = self.entropy[idx]
        label = self.labels[idx]

        if self.augment:
            seq = self._augment(seq)

        return seq, ent, label

    def _augment(self, seq: torch.Tensor) -> torch.Tensor:
        """Light augmentation: temporal masking + jitter."""
        L = seq.shape[0]

        # Random temporal masking (mask 5 to 15% of timesteps)
        if torch.rand(1).item() < 0.3:
            mask_len = int(L * (0.05 + 0.10 * torch.rand(1).item()))
            start = torch.randint(0, max(1, L - mask_len), (1,)).item()
            seq[start:start + mask_len] = 0.0

        # Small Gaussian jitter on continuous features
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(seq[:, :2]) * 0.05
            seq[:, :2] = seq[:, :2] + noise

        return seq


# ── Helpers ───────────────────────────────────────────────────────────

def load_entropy_h5(h5_path: str):
    """Load HDF5 with sequences + entropy features + flow_ids."""
    with h5py.File(h5_path, "r") as hf:
        X_seq = hf["sequences"][:].astype(np.float32)
        X_ent = hf["entropy"][:].astype(np.float32)
        y = hf["labels"][:].astype(np.int64)
        n_ent = int(hf["metadata"].attrs.get(
            "n_entropy_features", X_ent.shape[1]))
        # Load flow IDs for flow-isolated splitting
        flow_ids = None
        if "flow_ids" in hf:
            raw = hf["flow_ids"][:]
            flow_ids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in raw]
    logger.info("Loaded %d flows (seq=%s, entropy=%s, flow_ids=%s) from %s",
                len(y), X_seq.shape, X_ent.shape,
                "yes" if flow_ids else "no", h5_path)
    return X_seq, X_ent, y, n_ent, flow_ids


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for WeightedRandomSampler."""
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    weights = np.zeros(len(y), dtype=np.float64)
    for c, w in zip(classes, weight_per_class):
        weights[y == c] = w
    return weights


def evaluate(model, loader, device, num_classes, use_entropy=True):
    """Evaluate model on a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            seq, ent, lab = batch
            seq = seq.to(device)
            ent = ent.to(device) if use_entropy else None
            logits = model(seq, entropy_features=ent)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lab.numpy())

    return compute_metrics(np.array(all_labels), np.array(all_preds), num_classes)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train BiMamba + Entropy on packet sequences")

    # Data
    parser.add_argument("--data", type=str,
                        default="data/sequences/entropy_flows.h5")
    parser.add_argument("--num-classes", type=int, default=8)

    # Architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--unidirectional", action="store_true",
                        help="Use only the forward scan (ablates the "
                             "backward direction)")
    parser.add_argument("--no-entropy", action="store_true",
                        help="Shorthand for --entropy-mode none")
    parser.add_argument("--entropy-mode", type=str, default=None,
                        choices=list(P.ENTROPY_MODES),
                        help="How the context channel is presented "
                             "(default: real)")
    parser.add_argument("--split", type=str, default="flow-isolated",
                        choices=list(P.SPLIT_MODES),
                        help="Splitting protocol (default: flow-isolated)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Truncate packet sequences to this length")
    parser.add_argument("--subsample-per-class", type=int, default=None,
                        help="Cap flows per class before splitting")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name for saved files (default: auto)")

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training augmentation")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N epochs (default: 50)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")

    # Output
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Device: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    entropy_mode = args.entropy_mode or ("none" if args.no_entropy else "real")
    args.entropy_mode = entropy_mode
    use_entropy = entropy_mode != "none"

    # Construct model name
    if args.name is None:
        args.name = (f"bimamba_d{args.d_model}_L{args.n_layers}_"
                     f"{entropy_mode}")
    logger.info("Run name: %s", args.name)
    logger.info("Protocol: split=%s entropy=%s augment=%s seed=%d",
                args.split, entropy_mode, args.augment, args.seed)

    # ── Load data ─────────────────────────────────────────────────────
    X_seq, X_ent, y, flow_ids = P.load_corpus(args.data, seq_len=args.seq_len)
    n_ent = X_ent.shape[1]
    logger.info("Loaded %d flows: seq=%s entropy=%s",
                len(y), X_seq.shape, X_ent.shape)

    if args.subsample_per_class:
        keep = P.stratified_subsample(y, args.subsample_per_class, args.seed)
        X_seq, X_ent, y, flow_ids = (X_seq[keep], X_ent[keep],
                                     y[keep], flow_ids[keep])
        logger.info("Subsampled to %d flows (cap %d per class)",
                    len(y), args.subsample_per_class)

    # Class distribution
    classes, counts = np.unique(y, return_counts=True)
    logger.info("Class distribution:")
    for c, cnt in zip(classes, counts):
        name = ID_TO_GROUP_LABEL.get(int(c), str(c))
        logger.info("  %s (%d): %d flows", name, c, cnt)

    # ── Split first, then fit every statistic on train only ───────────
    tr, va, te = P.split_indices(y, flow_ids, X_ent, args.split, args.seed)
    groups = P._group_keys(args.split, y, flow_ids, X_ent)
    split_meta = P.split_report(y, (tr, va, te), groups)
    logger.info("Train: %d, Val: %d, Test: %d (groups=%s, crossing=%s)",
                len(tr), len(va), len(te),
                split_meta.get("n_groups"),
                split_meta.get("groups_crossing_partitions"))

    # The context channel is rewritten before normalisation. Substitute
    # vectors are drawn from the training partition only.
    X_ent = P.apply_entropy_mode(
        X_ent, y, entropy_mode, args.seed,
        reference=X_ent[tr], reference_y=y[tr])

    norm = P.Normalizer.fit(X_seq[tr], X_ent[tr])
    X_seq = norm.apply_seq(X_seq)
    X_ent = norm.apply_ent(X_ent) if use_entropy else X_ent

    X_tr_s, X_tr_e, y_tr = X_seq[tr], X_ent[tr], y[tr]
    X_va_s, X_va_e, y_va = X_seq[va], X_ent[va], y[va]
    X_te_s, X_te_e, y_te = X_seq[te], X_ent[te], y[te]

    # Datasets
    train_ds = EntropyFlowDataset(X_tr_s, X_tr_e, y_tr, augment=args.augment)
    val_ds = EntropyFlowDataset(X_va_s, X_va_e, y_va, augment=False)
    test_ds = EntropyFlowDataset(X_te_s, X_te_e, y_te, augment=False)

    # Weighted sampler for class balance
    sample_weights = compute_sample_weights(y_tr)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = BiMambaController(
        num_classes=args.num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        input_dim=X_tr_s.shape[2],
        n_entropy_features=n_ent,
        use_entropy=use_entropy,
        bidirectional=not args.unidirectional,
    ).to(device)

    n_params = model.count_parameters()
    logger.info("BiMamba params: %d (%.2f M)", n_params, n_params / 1e6)
    logger.info("  d_model=%d, n_layers=%d (×2 bidirectional), entropy=%s",
                args.d_model, args.n_layers, use_entropy)

    if args.compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    # ── Loss + Optimizer ──────────────────────────────────────
    # Class weights are estimated on the training partition only.
    tr_classes, tr_counts = np.unique(y_tr, return_counts=True)
    alpha = torch.zeros(args.num_classes, dtype=torch.float32)
    for c, cnt in zip(tr_classes, tr_counts):
        alpha[int(c)] = min(len(y_tr) / (len(tr_classes) * cnt), 50.0)
    criterion = FocalLoss(alpha=alpha.to(device), gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / \
            max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler("cuda", enabled=use_amp)
    logger.info("AMP: %s", "ON" if use_amp else "OFF")

    best_val_f1 = 0.0
    patience_counter = 0
    start_epoch = 1
    history = {"train_loss": [], "val_f1": [], "val_acc": [], "lr": []}

    # ── Resume from checkpoint ────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
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

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for seq, ent, lab in pbar:
            seq = seq.to(device)
            ent = ent.to(device) if use_entropy else None
            lab = lab.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                logits = model(seq, entropy_features=ent)
                loss = criterion(logits, lab)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        # Validate
        val_m = evaluate(model, val_loader, device,
                         args.num_classes, use_entropy)
        val_f1 = val_m["macro_f1"]
        val_acc = val_m["accuracy"]

        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %d/%d | loss=%.4f | val_acc=%.4f | val_f1=%.4f | "
            "lr=%.2e | %.1fs",
            epoch, args.epochs, avg_loss, val_acc, val_f1, current_lr, epoch_time)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(args.output_dir, f"{args.name}_best.pt")
            state = (model._orig_mod if hasattr(model, "_orig_mod")
                     else model).state_dict()
            torch.save(state, model_path)
            logger.info("  → New best val_f1=%.4f, saved!", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Periodic checkpoint
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"{args.name}_epoch{epoch}.pt")
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "patience_counter": patience_counter,
                "history": history,
                "args": vars(args),
            }
            if use_amp:
                ckpt_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
            logger.info("  Checkpoint saved: %s", ckpt_path)

    # ── Test evaluation ───────────────────────────────────────────────
    logger.info("Loading best model for test...")
    (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(
        torch.load(os.path.join(args.output_dir, f"{args.name}_best.pt"),
                   map_location=device, weights_only=True))

    test_m = evaluate(model, test_loader, device,
                      args.num_classes, use_entropy)
    class_names = [ID_TO_GROUP_LABEL.get(i, str(i))
                   for i in range(args.num_classes)]

    logger.info("=" * 60)
    logger.info("TEST RESULTS — %s (%s split)", args.name, args.split)
    logger.info("=" * 60)
    print_classification_report(test_m, class_names)

    # DDoS-DoS confusion analysis
    cm = test_m.get("confusion_matrix")
    if cm is not None and cm.shape[0] > 2:
        ddos_as_dos = cm[1, 2]
        dos_as_ddos = cm[2, 1]
        ddos_total = cm[1].sum()
        dos_total = cm[2].sum()
        logger.info("DDoS→DoS confusion: %d/%d (%.2f%%)",
                    ddos_as_dos, ddos_total,
                    100 * ddos_as_dos / max(ddos_total, 1))
        logger.info("DoS→DDoS confusion: %d/%d (%.2f%%)",
                    dos_as_ddos, dos_total,
                    100 * dos_as_ddos / max(dos_total, 1))

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "model": args.name,
        "architecture": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "d_state": args.d_state,
            "expand": args.expand,
            "seq_len": int(X_tr_s.shape[1]),
            "bidirectional": not args.unidirectional,
            "use_entropy": use_entropy,
            "n_entropy_features": n_ent,
            "n_params": n_params,
        },
        "protocol": {
            "split": args.split,
            "entropy_mode": entropy_mode,
            "augment": args.augment,
            "seed": args.seed,
            "subsample_per_class": args.subsample_per_class,
            "corpus": args.data,
            "normalisation": "train-partition only",
            "split_report": split_meta,
        },
        "training": {
            "epochs": epoch,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "augment": args.augment,
            "focal_gamma": args.focal_gamma,
            "split": args.split,
        },
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

    results_path = os.path.join(args.results_dir, f"{args.name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Save test data for downstream evaluation
    test_save_path = os.path.join(
        args.output_dir, f"{args.name}_test_data.npz")
    np.savez_compressed(
        test_save_path,
        X_seq=X_te_s, X_ent=X_te_e, y=y_te)
    logger.info("Test data saved to %s", test_save_path)


if __name__ == "__main__":
    main()
