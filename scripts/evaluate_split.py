"""
evaluate_split.py — Split-Inference evaluation (CNN data plane + BiMamba controller).

Implements the selective escalation architecture:
  1. DataPlaneCNN (tabular MLP, 39 features) classifies ALL flows
  2. Flows where CNN confidence margin < threshold → "ambiguous"
  3. Ambiguous flows → BiMamba+Entropy (control plane)
  4. Final prediction = CNN for confident, BiMamba for ambiguous

The CNN operates on tabular flow statistics (data/test.npz) and the
BiMamba operates on packet sequences + entropy (models/bimamba_ent_test_data.npz).
Since the two test sets represent different samples, combined performance
is estimated analytically.

Reports:
  - CNN-only accuracy on tabular test data
  - CNN confidence distribution and escalation analysis
  - BiMamba-only accuracy on sequence test data
  - Estimated combined split accuracy at various thresholds

Usage:
    python scripts/evaluate_split.py
    python scripts/evaluate_split.py --threshold 0.5
    python scripts/evaluate_split.py --device cpu

Requires:
    - models/cnn_best.pt                (DataPlaneCNN, tabular MLP)
    - models/bimamba_ent_best.pt        (BiMamba+Entropy)
    - data/test.npz                     (tabular test: X(N,39), y(N,))
    - models/bimamba_ent_test_data.npz  (sequence test: X_seq, X_ent, y)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402
from src.utils.metrics import compute_metrics, print_classification_report  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX inference helpers
# ---------------------------------------------------------------------------

def onnx_cnn_predict_with_confidence(session, X_tab, batch_size=512):
    """Run CNN ONNX model and return predictions, confidence margins, entropy."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    all_preds, all_margins, all_entropy = [], [], []

    for start in range(0, len(X_tab), batch_size):
        batch = X_tab[start:start + batch_size].astype(np.float32)
        (logits,) = session.run([output_name], {input_name: batch})

        # Softmax
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        # Top-2 margin (matches DataPlaneCNN.predict_with_confidence)
        top2_idx = np.argsort(probs, axis=1)[:, -2:]
        p_top1 = np.take_along_axis(probs, top2_idx[:, -1:], axis=1).squeeze(1)
        p_top2 = np.take_along_axis(
            probs, top2_idx[:, -2:-1], axis=1).squeeze(1)
        margin = p_top1 - p_top2

        # Predictive entropy
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

        all_preds.append(top2_idx[:, -1])
        all_margins.append(margin)
        all_entropy.append(entropy)

    return (np.concatenate(all_preds),
            np.concatenate(all_margins),
            np.concatenate(all_entropy))


def onnx_bimamba_predict(session, X_seq, X_ent, batch_size=256):
    """Run BiMamba ONNX model and return class predictions."""
    in_names = [inp.name for inp in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    all_preds = []

    for start in range(0, len(X_seq), batch_size):
        seq_b = X_seq[start:start + batch_size].astype(np.float32)
        ent_b = X_ent[start:start + batch_size].astype(np.float32)
        (logits,) = session.run([out_name],
                                {in_names[0]: seq_b, in_names[1]: ent_b})
        all_preds.append(logits.argmax(axis=1))

    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# PyTorch inference helpers
# ---------------------------------------------------------------------------

def _load_torch():
    """Lazy-import torch (not needed in --onnx mode)."""
    import torch  # noqa: E402
    return torch


def cnn_predict_with_confidence(model, X_tab, device, batch_size=512):
    """Run tabular DataPlaneCNN and return predictions, confidence margins, entropy."""
    torch = _load_torch()
    model.eval()
    all_preds, all_margins, all_entropy = [], [], []

    with torch.no_grad():
        for start in range(0, len(X_tab), batch_size):
            batch = torch.from_numpy(
                X_tab[start:start + batch_size]).float().to(device)
            pred, margin, _, entropy = model.predict_with_confidence(batch)
            all_preds.append(pred.cpu().numpy())
            all_margins.append(margin.cpu().numpy())
            all_entropy.append(entropy.cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_margins),
            np.concatenate(all_entropy))


def bimamba_predict(model, X_seq, X_ent, device, batch_size=256):
    """Run BiMamba+Entropy on sequence + entropy data."""
    torch = _load_torch()
    model.eval()
    all_preds = []

    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            seq_batch = torch.from_numpy(
                X_seq[start:start + batch_size]).float().to(device)
            ent_batch = torch.from_numpy(
                X_ent[start:start + batch_size]).float().to(device)
            logits = model(seq_batch, ent_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# ONNX evaluation mode
# ---------------------------------------------------------------------------

def _load_data(args):
    """Load and normalize test data for both CNN and BiMamba."""
    # Tabular test (CNN)
    logger.info("Loading tabular test data from %s", args.tabular_test)
    tab_data = np.load(args.tabular_test)
    X_tab = tab_data["X"].astype(np.float32)
    y_tab = tab_data["y"].astype(np.int64)

    norm_path = os.path.join(os.path.dirname(args.tabular_test),
                             "norm_stats.npz")
    if os.path.exists(norm_path):
        norms = np.load(norm_path)
        X_tab = (X_tab - norms["mean"]) / np.clip(norms["std"], 1e-8, None)
        logger.info("Applied z-score normalization from %s", norm_path)

    # Sequence test (BiMamba)
    logger.info("Loading sequence test data from %s", args.seq_test)
    seq_data = np.load(args.seq_test)
    X_seq = seq_data["X_seq"].astype(np.float32)
    X_ent = seq_data["X_ent"].astype(np.float32)
    y_seq = seq_data["y"].astype(np.int64)

    logger.info("Tabular: %d samples x %d feats | Sequence: %d samples",
                len(y_tab), X_tab.shape[1], len(y_seq))
    return X_tab, y_tab, X_seq, X_ent, y_seq


def run_onnx_evaluation(args, class_names):
    """Run split-inference evaluation using ONNX Runtime (deployment mode)."""
    import onnxruntime as ort

    X_tab, y_tab, X_seq, X_ent, y_seq = _load_data(args)

    # Load ONNX models
    logger.info("Loading CNN ONNX from %s", args.cnn_onnx)
    cnn_sess = ort.InferenceSession(
        args.cnn_onnx, providers=["CPUExecutionProvider"])

    logger.info("Loading BiMamba ONNX from %s", args.bimamba_onnx)
    bimamba_sess = ort.InferenceSession(
        args.bimamba_onnx, providers=["CPUExecutionProvider"])

    # ---- CNN predictions (data plane) ----
    logger.info("Running CNN ONNX inference on tabular test data...")
    t0 = time.time()
    cnn_preds, cnn_margin, cnn_entropy = onnx_cnn_predict_with_confidence(
        cnn_sess, X_tab)
    cnn_time = time.time() - t0

    cnn_metrics = compute_metrics(y_tab, cnn_preds, args.num_classes)
    logger.info("=== CNN ONNX (Data Plane, Tabular) ===")
    logger.info("  Accuracy: %.4f | Macro F1: %.4f | Time: %.3f s (%.1f us/sample)",
                cnn_metrics["accuracy"], cnn_metrics["macro_f1"],
                cnn_time, 1e6 * cnn_time / len(y_tab))

    # ---- BiMamba predictions (control plane) ----
    logger.info("Running BiMamba ONNX inference on sequence test data...")
    t0 = time.time()
    bimamba_preds = onnx_bimamba_predict(bimamba_sess, X_seq, X_ent)
    bimamba_time = time.time() - t0

    bimamba_metrics = compute_metrics(y_seq, bimamba_preds, args.num_classes)
    logger.info("=== BiMamba ONNX (Control Plane, Sequences) ===")
    logger.info("  Accuracy: %.4f | Macro F1: %.4f | Time: %.3f s (%.1f us/sample)",
                bimamba_metrics["accuracy"], bimamba_metrics["macro_f1"],
                bimamba_time, 1e6 * bimamba_time / len(y_seq))

    # ---- Threshold sweep + split-inference estimate ----
    bimamba_acc = bimamba_metrics["accuracy"]
    bimamba_f1 = bimamba_metrics["macro_f1"]
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = []

    logger.info("\n=== ONNX Split-Inference Estimate ===")
    logger.info("  (BiMamba standalone: acc=%.4f, macro_f1=%.4f)",
                bimamba_acc, bimamba_f1)
    logger.info("%-10s  %-8s  %-10s  %-10s  %-12s",
                "Threshold", "CNN %", "CNN Acc", "Escalated %",
                "Est.Comb.Acc")
    logger.info("-" * 60)

    for thr in thresholds:
        confident_mask = cnn_margin >= thr
        n_confident = confident_mask.sum()
        pct_cnn = 100.0 * n_confident / len(y_tab)
        pct_esc = 100.0 - pct_cnn

        cnn_conf_acc = (
            (cnn_preds[confident_mask] == y_tab[confident_mask]).mean()
            if n_confident > 0 else 0.0)

        est_combined_acc = (
            (pct_cnn / 100.0) * cnn_conf_acc +
            (pct_esc / 100.0) * bimamba_acc)

        logger.info(
            "%-10.2f  %-8.1f  %-10.4f  %-10.1f  %-12.4f",
            thr, pct_cnn, cnn_conf_acc, pct_esc, est_combined_acc)

        sweep_results.append({
            "threshold": thr,
            "pct_cnn": round(float(pct_cnn), 2),
            "n_confident": int(n_confident),
            "cnn_confident_accuracy": round(float(cnn_conf_acc), 4),
            "escalated_pct": round(float(pct_esc), 2),
            "estimated_combined_accuracy": round(float(est_combined_acc), 4),
        })

    # ---- Detailed report at default threshold ----
    thr = args.threshold
    confident_mask = cnn_margin >= thr

    logger.info("\n=== Detailed ONNX Report (threshold=%.2f) ===", thr)
    logger.info("CNN handles: %d / %d (%.1f%%) flows on data plane",
                confident_mask.sum(), len(y_tab),
                100 * confident_mask.sum() / len(y_tab))
    logger.info("Escalated to BiMamba: %d flows", (~confident_mask).sum())

    logger.info("\n--- CNN ONNX classification (all flows) ---")
    print_classification_report(cnn_metrics, class_names)

    logger.info("\n--- BiMamba ONNX classification (sequence test) ---")
    print_classification_report(bimamba_metrics, class_names)

    # ---- Per-class escalation ----
    logger.info("\n=== Per-Class Escalation Analysis (threshold=%.2f) ===", thr)
    logger.info("%-15s  %-8s  %-8s  %-10s  %-12s",
                "Class", "Total", "Escal", "Esc %", "CNN Acc")
    for c in range(args.num_classes):
        c_mask = y_tab == c
        c_total = c_mask.sum()
        c_esc = (c_mask & ~confident_mask).sum()
        c_pct = 100.0 * c_esc / max(c_total, 1)
        c_cnn_acc = (
            (cnn_preds[c_mask] == y_tab[c_mask]).mean()
            if c_total > 0 else 0.0)
        logger.info("%-15s  %-8d  %-8d  %-10.1f  %-12.4f",
                    class_names[c], c_total, c_esc, c_pct, c_cnn_acc)

    # ---- Best operating point ----
    best = max(sweep_results,
               key=lambda x: x["estimated_combined_accuracy"])
    logger.info("\n=== Best Operating Point ===")
    logger.info("  Threshold: %.2f", best["threshold"])
    logger.info("  Est. Combined Accuracy: %.4f",
                best["estimated_combined_accuracy"])
    logger.info("  CNN handles: %.1f%% (fast path)", best["pct_cnn"])
    logger.info("  Escalated: %.1f%% (slow path)", best["escalated_pct"])

    # ---- Save results ----
    results = {
        "mode": "onnx",
        "cnn_onnx": {
            "model": args.cnn_onnx,
            "accuracy": round(cnn_metrics["accuracy"], 4),
            "macro_f1": round(cnn_metrics["macro_f1"], 4),
            "weighted_f1": round(cnn_metrics["weighted_f1"], 4),
            "inference_time_s": round(cnn_time, 3),
            "us_per_sample": round(1e6 * cnn_time / len(y_tab), 1),
            "n_samples": len(y_tab),
        },
        "bimamba_onnx": {
            "model": args.bimamba_onnx,
            "accuracy": round(bimamba_metrics["accuracy"], 4),
            "macro_f1": round(bimamba_metrics["macro_f1"], 4),
            "weighted_f1": round(bimamba_metrics["weighted_f1"], 4),
            "inference_time_s": round(bimamba_time, 3),
            "us_per_sample": round(1e6 * bimamba_time / len(y_seq), 1),
            "n_samples": len(y_seq),
        },
        "best_operating_point": best,
        "threshold_sweep": sweep_results,
        "per_class_escalation": [
            {
                "class": class_names[c],
                "total": int((y_tab == c).sum()),
                "escalated": int(
                    ((y_tab == c) & (cnn_margin < best["threshold"])).sum()),
            }
            for c in range(args.num_classes)
        ],
    }

    results_path = os.path.join(args.results_dir,
                                "split_evaluation_onnx.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("ONNX results saved to %s", results_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Split-Inference (tabular CNN + BiMamba)")
    parser.add_argument("--tabular-test", type=str, default="data/test.npz",
                        help="Tabular test data for CNN (X: N x 39, y: N)")
    parser.add_argument("--seq-test", type=str,
                        default="models/bimamba_ent_test_data.npz",
                        help="Sequence test data for BiMamba (X_seq, X_ent, y)")
    parser.add_argument("--cnn-model", type=str, default="models/cnn_best.pt",
                        help="Tabular CNN checkpoint")
    parser.add_argument("--bimamba-model", type=str,
                        default="models/bimamba_ent_best.pt")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Default confidence threshold for detailed report")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--onnx", action="store_true",
                        help="Use ONNX Runtime instead of PyTorch (deployment mode)")
    parser.add_argument("--cnn-onnx", type=str,
                        default="models/quantized/cnn_quantized.onnx",
                        help="ONNX CNN model (used with --onnx)")
    parser.add_argument("--bimamba-onnx", type=str,
                        default="models/onnx/bimamba_ent.onnx",
                        help="ONNX BiMamba model (used with --onnx)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    class_names = [ID_TO_GROUP_LABEL.get(i, str(i))
                   for i in range(args.num_classes)]

    if args.onnx:
        return run_onnx_evaluation(args, class_names)

    torch = _load_torch()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    # ---- Load tabular test data (for CNN) ----
    logger.info("Loading tabular test data from %s", args.tabular_test)
    tab_data = np.load(args.tabular_test)
    X_tab = tab_data["X"].astype(np.float32)
    y_tab = tab_data["y"].astype(np.int64)
    n_features = X_tab.shape[1]

    # Apply z-score normalization (must match training)
    norm_path = os.path.join(os.path.dirname(
        args.tabular_test), "norm_stats.npz")
    if os.path.exists(norm_path):
        norms = np.load(norm_path)
        X_tab = (X_tab - norms["mean"]) / np.clip(norms["std"], 1e-8, None)
        logger.info("Applied z-score normalization from %s", norm_path)
    else:
        logger.warning("No norm_stats.npz found; CNN accuracy may be degraded")

    logger.info("Tabular test: %d samples, %d features",
                len(y_tab), n_features)

    # ---- Load sequence test data (for BiMamba) ----
    logger.info("Loading sequence test data from %s", args.seq_test)
    seq_data = np.load(args.seq_test)
    X_seq = seq_data["X_seq"].astype(np.float32)
    X_ent = seq_data["X_ent"].astype(np.float32)
    y_seq = seq_data["y"].astype(np.int64)
    logger.info("Sequence test: %d samples (seq: %s, ent: %s)",
                len(y_seq), X_seq.shape, X_ent.shape)

    # ---- Load DataPlaneCNN (tabular MLP) ----
    from src.models.data_plane_cnn import DataPlaneCNN  # noqa: E402
    from src.models.bimamba_controller import BiMambaController  # noqa: E402

    logger.info("Loading tabular CNN from %s", args.cnn_model)
    cnn = DataPlaneCNN(
        num_classes=args.num_classes,
        input_features=n_features).to(device)

    if os.path.exists(args.cnn_model):
        ckpt = torch.load(
            args.cnn_model, map_location=device, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        cnn.load_state_dict(sd)
    else:
        logger.error("CNN checkpoint not found: %s", args.cnn_model)
        return

    # ---- Load BiMamba+Entropy ----
    logger.info("Loading BiMamba from %s", args.bimamba_model)
    bimamba = BiMambaController(
        num_classes=args.num_classes,
        d_model=64, n_layers=4,
        d_state=16, d_conv=4, expand=1,
        n_entropy_features=6, use_entropy=True,
        input_dim=3, dropout=0.15,
    ).to(device)
    ckpt = torch.load(args.bimamba_model,
                      map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    bimamba.load_state_dict(sd)
    bimamba.eval()

    # ---- CNN predictions (data plane, tabular) ----
    logger.info("Running CNN data-plane predictions on tabular test data…")
    t0 = time.time()
    cnn_preds, cnn_margin, cnn_entropy = cnn_predict_with_confidence(
        cnn, X_tab, device)
    cnn_time = time.time() - t0

    cnn_metrics = compute_metrics(y_tab, cnn_preds, args.num_classes)
    logger.info("=== CNN-Only (Data Plane, Tabular) ===")
    logger.info("  Accuracy: %.4f | Macro F1: %.4f | Time: %.3f s (%.1f us/sample)",
                cnn_metrics["accuracy"], cnn_metrics["macro_f1"],
                cnn_time, 1e6 * cnn_time / len(y_tab))

    # ---- BiMamba predictions (control plane, sequences) ----
    logger.info(
        "Running BiMamba control-plane predictions on sequence test data…")
    t0 = time.time()
    bimamba_preds = bimamba_predict(bimamba, X_seq, X_ent, device)
    bimamba_time = time.time() - t0

    bimamba_metrics = compute_metrics(y_seq, bimamba_preds, args.num_classes)
    logger.info("=== BiMamba-Only (Control Plane, Sequences) ===")
    logger.info("  Accuracy: %.4f | Macro F1: %.4f | Time: %.3f s (%.1f us/sample)",
                bimamba_metrics["accuracy"], bimamba_metrics["macro_f1"],
                bimamba_time, 1e6 * bimamba_time / len(y_seq))

    # ---- Threshold sweep (CNN confidence analysis) ----
    # Since tabular and sequence test sets are disjoint, we estimate
    # combined split-inference performance analytically:
    #   combined_acc = pct_confident * cnn_conf_acc + pct_escalated * bimamba_acc
    bimamba_acc = bimamba_metrics["accuracy"]
    bimamba_f1 = bimamba_metrics["macro_f1"]

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = []

    logger.info("\n=== CNN Confidence Analysis + Split-Inference Estimate ===")
    logger.info("  (BiMamba standalone: acc=%.4f, macro_f1=%.4f)",
                bimamba_acc, bimamba_f1)
    logger.info("%-10s  %-8s  %-10s  %-10s  %-12s",
                "Threshold", "CNN %", "CNN Acc", "Escalated %",
                "Est.Comb.Acc")
    logger.info("-" * 60)

    for thr in thresholds:
        confident_mask = cnn_margin >= thr
        ambiguous_mask = ~confident_mask

        n_confident = confident_mask.sum()
        n_ambiguous = ambiguous_mask.sum()
        pct_cnn = 100.0 * n_confident / len(y_tab)
        pct_esc = 100.0 * n_ambiguous / len(y_tab)

        # CNN accuracy on confident subset
        cnn_conf_acc = (
            (cnn_preds[confident_mask] == y_tab[confident_mask]).mean()
            if n_confident > 0 else 0.0)

        # Estimated combined accuracy (analytical)
        est_combined_acc = (
            (pct_cnn / 100.0) * cnn_conf_acc +
            (pct_esc / 100.0) * bimamba_acc)

        logger.info(
            "%-10.2f  %-8.1f  %-10.4f  %-10.1f  %-12.4f",
            thr, pct_cnn, cnn_conf_acc, pct_esc, est_combined_acc)

        sweep_results.append({
            "threshold": thr,
            "pct_cnn": round(float(pct_cnn), 2),
            "n_confident": int(n_confident),
            "n_ambiguous": int(n_ambiguous),
            "cnn_confident_accuracy": round(float(cnn_conf_acc), 4),
            "escalated_pct": round(float(pct_esc), 2),
            "estimated_combined_accuracy": round(float(est_combined_acc), 4),
        })

    # ---- Detailed report at default threshold ----
    thr = args.threshold
    confident_mask = cnn_margin >= thr

    logger.info("\n=== Detailed Report (threshold=%.2f) ===", thr)
    logger.info("CNN handles: %d / %d (%.1f%%) flows on data plane",
                confident_mask.sum(), len(y_tab),
                100 * confident_mask.sum() / len(y_tab))
    logger.info("Escalated to BiMamba: %d flows", (~confident_mask).sum())

    # CNN-only classification report on full tabular test
    logger.info("\n--- CNN-only classification (all flows) ---")
    print_classification_report(cnn_metrics, class_names)

    # BiMamba-only classification report on sequence test
    logger.info("\n--- BiMamba-only classification (sequence test) ---")
    print_classification_report(bimamba_metrics, class_names)

    # ---- Per-class CNN confidence analysis ----
    logger.info("\n=== Per-Class Escalation Analysis (threshold=%.2f) ===", thr)
    logger.info("%-15s  %-8s  %-8s  %-10s  %-12s",
                "Class", "Total", "Escal", "Esc %", "CNN Acc")
    for c in range(args.num_classes):
        c_mask = y_tab == c
        c_total = c_mask.sum()
        c_esc = (c_mask & ~confident_mask).sum()
        c_pct = 100.0 * c_esc / max(c_total, 1)
        c_cnn_acc = (
            (cnn_preds[c_mask] == y_tab[c_mask]).mean()
            if c_total > 0 else 0.0)
        logger.info("%-15s  %-8d  %-8d  %-10.1f  %-12.4f",
                    class_names[c], c_total, c_esc, c_pct, c_cnn_acc)

    # ---- Best operating point ----
    best = max(sweep_results,
               key=lambda x: x["estimated_combined_accuracy"])
    logger.info("\n=== Best Operating Point ===")
    logger.info("  Threshold: %.2f", best["threshold"])
    logger.info("  Est. Combined Accuracy: %.4f",
                best["estimated_combined_accuracy"])
    logger.info("  CNN handles: %.1f%% (fast path)", best["pct_cnn"])
    logger.info("  Escalated: %.1f%% (slow path)", best["escalated_pct"])

    # ---- Save results ----
    results = {
        "cnn_only": {
            "accuracy": round(cnn_metrics["accuracy"], 4),
            "macro_f1": round(cnn_metrics["macro_f1"], 4),
            "weighted_f1": round(cnn_metrics["weighted_f1"], 4),
            "inference_time_s": round(cnn_time, 3),
            "n_samples": len(y_tab),
            "n_features": n_features,
        },
        "bimamba_only": {
            "accuracy": round(bimamba_metrics["accuracy"], 4),
            "macro_f1": round(bimamba_metrics["macro_f1"], 4),
            "weighted_f1": round(bimamba_metrics["weighted_f1"], 4),
            "inference_time_s": round(bimamba_time, 3),
            "n_samples": len(y_seq),
        },
        "best_operating_point": best,
        "threshold_sweep": sweep_results,
        "per_class_escalation": [
            {
                "class": class_names[c],
                "total": int((y_tab == c).sum()),
                "escalated": int(
                    ((y_tab == c) & (cnn_margin < best["threshold"])).sum()),
            }
            for c in range(args.num_classes)
        ],
    }

    results_path = os.path.join(args.results_dir, "split_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
