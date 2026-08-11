"""
evaluate_models.py — Evaluate both CNN and Mamba models, including the
split-inference early-exit simulation.

Usage:
    python scripts/evaluate_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402
from src.utils.visualization import plot_confusion_matrix  # noqa: E402
from src.utils.metrics import compute_metrics, print_classification_report  # noqa: E402
from src.models.mamba_controller import MambaController  # noqa: E402
from src.models.data_plane_cnn import DataPlaneCNN  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def reshape_for_model(X, seq_len=100):
    """Legacy reshape — kept for reference but no longer used."""
    raise NotImplementedError("Use tabular mode for CNN, unsqueeze for Mamba")


def split_inference_simulation(
    cnn: DataPlaneCNN,
    mamba: MambaController,
    X_flat: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
):
    """
    Simulate the Split-Mamba early-exit pipeline:
      1. CNN classifies all flows (tabular mode — flat features)
      2. High-confidence flows → CNN prediction
      3. Low-confidence flows → Mamba prediction (unsqueezed features)
    """
    cnn.eval()
    mamba.eval()

    loader = DataLoader(TensorDataset(
        X_flat, y), batch_size=batch_size, shuffle=False)

    final_preds = []
    all_labels = []
    cnn_handled = 0
    mamba_handled = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)

        # CNN inference (flat features)
        pred_class, margin, _, entropy = cnn.predict_with_confidence(X_batch)

        # Entropy-based thresholding: H(p) = -sum(p_i * log(p_i))
        # Low entropy → confident → keep CNN prediction
        # High entropy → ambiguous → escalate to Mamba
        high_conf = entropy < threshold  # Note: threshold is now entropy-based
        low_conf = ~high_conf

        batch_preds = torch.zeros(X_batch.size(0), dtype=torch.long)

        # High confidence → use CNN prediction
        if high_conf.any():
            batch_preds[high_conf] = pred_class[high_conf].cpu()
            cnn_handled += high_conf.sum().item()

        # Low confidence → use Mamba prediction (unsqueeze for sequence)
        if low_conf.any():
            with torch.no_grad():
                X_mamba = X_batch[low_conf].unsqueeze(-1)  # (N, feat, 1)
                mamba_logits = mamba(X_mamba)
                mamba_preds = mamba_logits.argmax(dim=1).cpu()
            batch_preds[low_conf] = mamba_preds
            mamba_handled += low_conf.sum().item()

        final_preds.extend(batch_preds.numpy())
        all_labels.extend(y_batch.numpy())

    total = cnn_handled + mamba_handled
    logger.info("Split inference: CNN handled %d (%.1f%%), Mamba handled %d (%.1f%%)",
                cnn_handled, 100 * cnn_handled / max(total, 1),
                mamba_handled, 100 * mamba_handled / max(total, 1))

    return np.array(all_labels), np.array(final_preds), cnn_handled, mamba_handled


def main():
    parser = argparse.ArgumentParser(description="Evaluate Split-Mamba system")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--cnn-model", type=str, default="models/cnn_best.pt")
    parser.add_argument("--mamba-model", type=str,
                        default="models/mamba_best.pt")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for early exit")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    class_names = [ID_TO_GROUP_LABEL.get(i, str(i))
                   for i in range(args.num_classes)]

    # Load test data
    test_data = np.load(os.path.join(args.data_dir, "test.npz"))
    X_test = torch.from_numpy(test_data["X"]).float()
    y_test = torch.from_numpy(test_data["y"]).long()

    # Normalize using saved stats
    norm_path = os.path.join(args.data_dir, "norm_stats.npz")
    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        train_mean = torch.from_numpy(stats["mean"]).float()
        train_std = torch.from_numpy(stats["std"]).float()
        X_test = (X_test - train_mean) / train_std
    else:
        logger.warning("norm_stats.npz not found — using raw features")

    n_features = X_test.shape[1]

    os.makedirs(args.output_dir, exist_ok=True)

    # --- CNN-only evaluation ---
    if os.path.exists(args.cnn_model):
        logger.info("=== CNN-Only Evaluation ===")
        cnn = DataPlaneCNN(
            args.num_classes, input_features=n_features).to(device)
        cnn.load_state_dict(torch.load(
            args.cnn_model, map_location=device, weights_only=True))
        cnn.eval()

        loader = DataLoader(TensorDataset(X_test, y_test),
                            batch_size=256, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds = cnn(xb.to(device)).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        cnn_metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), args.num_classes)
        print_classification_report(cnn_metrics, class_names)
        plot_confusion_matrix(
            cnn_metrics["confusion_matrix"], class_names,
            title="CNN Confusion Matrix",
            save_path=os.path.join(args.output_dir, "cnn_confusion.png"),
        )
    else:
        cnn = None
        logger.warning("CNN model not found: %s", args.cnn_model)

    # --- Mamba-only evaluation ---
    if os.path.exists(args.mamba_model):
        logger.info("\n=== Mamba-Only Evaluation ===")
        mamba = MambaController(
            args.num_classes, d_model=256, n_layers=6,
            input_dim=1, sequence_length=n_features,
        ).to(device)
        mamba.load_state_dict(torch.load(
            args.mamba_model, map_location=device, weights_only=True))
        mamba.eval()

        X_test_seq = X_test.unsqueeze(-1)  # (batch, n_features, 1)
        loader = DataLoader(TensorDataset(X_test_seq, y_test),
                            batch_size=256, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds = mamba(xb.to(device)).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        mamba_metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), args.num_classes)
        print_classification_report(mamba_metrics, class_names)
        plot_confusion_matrix(
            mamba_metrics["confusion_matrix"], class_names,
            title="Mamba Confusion Matrix",
            save_path=os.path.join(args.output_dir, "mamba_confusion.png"),
        )
    else:
        mamba = None
        logger.warning("Mamba model not found: %s", args.mamba_model)

    # --- Split-inference evaluation ---
    if cnn is not None and mamba is not None:
        logger.info("\n=== Split-Mamba (Early-Exit) Evaluation ===")
        logger.info("Using entropy thresholding: H(p) = -sum(p_i * log(p_i))")
        logger.info("Max entropy for %d classes: %.3f",
                    args.num_classes, np.log(args.num_classes))
        for thresh in [0.3, 0.5, 1.0]:
            logger.info("--- Threshold: %.1f ---", thresh)
            y_true, y_pred, cnn_n, mamba_n = split_inference_simulation(
                cnn, mamba, X_test, y_test,
                threshold=thresh, device=device,
            )
            split_metrics = compute_metrics(y_true, y_pred, args.num_classes)
            print_classification_report(split_metrics, class_names)
            plot_confusion_matrix(
                split_metrics["confusion_matrix"], class_names,
                title=f"Split-Mamba (θ={thresh})",
                save_path=os.path.join(
                    args.output_dir, f"split_confusion_t{thresh}.png"),
            )

    logger.info("Evaluation complete. Results in %s/", args.output_dir)


if __name__ == "__main__":
    main()
