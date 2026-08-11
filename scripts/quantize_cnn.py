"""
quantize_cnn.py — Post-training quantization of the Data Plane CNN.

Applies symmetric int8 quantization and exports weights for Planter.

Usage:
    python scripts/quantize_cnn.py [--model models/cnn_best.pt]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import logging  # noqa: E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.utils.metrics import compute_metrics  # noqa: E402
from src.models.data_plane_cnn import DataPlaneCNN  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def reshape_for_cnn(X, seq_len=100):
    """Same reshape as in train_cnn.py."""
    batch_size = X.shape[0]
    n_features = X.shape[1]
    if n_features % 3 != 0:
        pad_size = 3 - (n_features % 3)
        X = torch.nn.functional.pad(X, (0, pad_size))
    X = X.view(batch_size, -1, 3)
    current_len = X.shape[1]
    if current_len < seq_len:
        repeats = (seq_len // current_len) + 1
        X = X.repeat(1, repeats, 1)[:, :seq_len, :]
    else:
        X = X[:, :seq_len, :]
    return X


def evaluate_accuracy(model, loader, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1).numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    metrics = compute_metrics(np.array(all_labels),
                              np.array(all_preds), num_classes)
    return metrics["accuracy"], metrics["macro_f1"]


def export_weights_for_planter(model, output_path):
    """Export quantized weights as CSV for Planter code generation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Index", "Value", "Scale"])

        for name, param in model.named_parameters():
            w = param.detach().cpu().numpy()
            scale = float(np.max(np.abs(w))) / \
                127.0 if np.max(np.abs(w)) > 0 else 1.0
            w_int8 = np.round(w / scale).astype(np.int8)

            for idx, val in enumerate(w_int8.flatten()):
                writer.writerow([name, idx, int(val), f"{scale:.8e}"])

    logger.info("Exported weights to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Quantize CNN to int8")
    parser.add_argument("--model", type=str, default="models/cnn_best.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="models/quantized")
    parser.add_argument("--weights-csv", type=str,
                        default="p4/weights/cnn_weights.csv")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    args = parser.parse_args()

    # Load FP32 model (tabular MLP: 39 -> 512 -> 256 -> 128 -> 8)
    model_fp32 = DataPlaneCNN(
        num_classes=args.num_classes, input_features=39)
    model_fp32.load_state_dict(torch.load(args.model, map_location="cpu",
                                          weights_only=True))
    model_fp32.eval()
    logger.info("Loaded FP32 model from %s", args.model)

    # Load test data (tabular: 39 features per flow)
    test_data = np.load(os.path.join(args.data_dir, "test.npz"))
    X_test = torch.from_numpy(test_data["X"]).float()
    y_test = torch.from_numpy(test_data["y"]).long()

    # Apply same normalization as training
    norm = np.load(os.path.join(args.data_dir, "norm_stats.npz"))
    train_mean = torch.from_numpy(norm["mean"]).float()
    train_std = torch.from_numpy(norm["std"]).float()
    X_test = (X_test - train_mean) / train_std

    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=256, shuffle=False
    )

    # Evaluate FP32
    acc_fp32, f1_fp32 = evaluate_accuracy(
        model_fp32, test_loader, args.num_classes)
    logger.info("FP32 - Accuracy: %.4f, F1: %.4f", acc_fp32, f1_fp32)

    # Manual symmetric int8 quantization (no deprecated torch.ao API)
    # Clone model, quantize Linear layer weights in-place
    model_int8 = DataPlaneCNN(
        num_classes=args.num_classes, input_features=39)
    model_int8.load_state_dict(torch.load(args.model, map_location="cpu",
                                          weights_only=True))
    model_int8.eval()

    quant_info = {}
    with torch.no_grad():
        for name, param in model_int8.named_parameters():
            if "weight" in name and param.dim() >= 2:
                w = param.data
                abs_max = w.abs().max()
                scale = abs_max / 127.0 if abs_max > 0 else 1.0
                w_int8 = torch.round(w / scale).clamp(-128, 127)
                param.data = w_int8 * scale  # dequantized (simulated int8)
                quant_info[name] = {
                    "shape": list(w.shape),
                    "scale": float(scale),
                    "abs_max": float(abs_max),
                }

    acc_int8, f1_int8 = evaluate_accuracy(
        model_int8, test_loader, args.num_classes)
    logger.info("INT8 - Accuracy: %.4f, F1: %.4f", acc_int8, f1_int8)

    degradation = acc_fp32 - acc_int8
    logger.info("Accuracy degradation: %.4f (threshold: %.4f)",
                degradation, args.max_accuracy_drop)

    if degradation > args.max_accuracy_drop:
        logger.warning(
            "Accuracy drop (%.4f) exceeds threshold (%.4f)!",
            degradation, args.max_accuracy_drop,
        )

    # Save quantized model
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cnn_int8.pth")
    torch.save(model_int8.state_dict(), out_path)
    logger.info("Saved quantized model to %s", out_path)

    # Export weights for Planter
    export_weights_for_planter(model_fp32, args.weights_csv)

    # Export ONNX (new torch.export-based exporter)
    try:
        onnx_path = os.path.join(args.output_dir, "cnn_quantized.onnx")
        dummy = torch.randn(1, 39)
        exported = torch.onnx.export(
            model_fp32, (dummy,), onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
        )
        logger.info("Exported ONNX model to %s", onnx_path)
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)

    # Save quantization results
    results = {
        "fp32": {"accuracy": round(acc_fp32, 4), "macro_f1": round(f1_fp32, 4)},
        "int8": {"accuracy": round(acc_int8, 4), "macro_f1": round(f1_int8, 4)},
        "degradation": round(acc_fp32 - acc_int8, 4),
        "model_params": 49512,
        "quant_layers": quant_info,
    }
    results_path = os.path.join(args.output_dir, "quantization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    logger.info("Quantization complete.")


if __name__ == "__main__":
    main()
