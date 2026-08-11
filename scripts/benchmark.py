"""
benchmark.py — Measure inference latency, throughput, and model size
for all trained models (data-plane CNN, Seq-CNN, BiMamba, ONNX).

Usage:
    python scripts/benchmark.py [--device cuda]
    python scripts/benchmark.py --device cpu   # CPU-only numbers

Reports:
    - Inference time (ms) per sample and per batch
    - Throughput (samples/sec)
    - Model size (parameters, MB)
    - Split-inference simulation with confidence-based escalation
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.mamba_controller import MambaController  # noqa: E402
from src.models.data_plane_cnn import DataPlaneCNN  # noqa: E402
from src.models.bimamba_controller import BiMambaController  # noqa: E402


# SequenceCNN: inline definition to avoid heavy import chain from train_seq_cnn
class SequenceCNN(torch.nn.Module):
    """1D-CNN for per-packet flow classification (from train_seq_cnn.py)."""

    def __init__(self, num_classes: int = 8, seq_len: int = 100):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        return self.classifier(x)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size()
                  for p in model.parameters()) / 1e6
    return {"total": total, "trainable": trainable, "size_mb": round(size_mb, 3)}


def benchmark_latency(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iterations: int = 200,
    entropy_shape: tuple = None,
) -> dict:
    """
    Measure single-sample and batch inference latency.

    Returns dict with latency_ms_per_sample, throughput_samples_sec, etc.
    """
    model.eval()
    model.to(device)

    has_entropy = entropy_shape is not None

    # --- Single-sample latency ---
    x_single = torch.randn(1, *input_shape, device=device)
    ent_single = (torch.randn(1, *entropy_shape, device=device)
                  if has_entropy else None)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if has_entropy:
                _ = model(x_single, ent_single)
            else:
                _ = model(x_single)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if has_entropy:
                _ = model(x_single, ent_single)
            else:
                _ = model(x_single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    single_ms = np.median(times)
    single_p99 = np.percentile(times, 99)

    # --- Batch latency (batch=64) ---
    batch_size = 64
    x_batch = torch.randn(batch_size, *input_shape, device=device)
    ent_batch = (torch.randn(batch_size, *entropy_shape, device=device)
                 if has_entropy else None)

    with torch.no_grad():
        for _ in range(warmup // 2):
            if has_entropy:
                _ = model(x_batch, ent_batch)
            else:
                _ = model(x_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    batch_times = []
    for _ in range(iterations // 2):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if has_entropy:
                _ = model(x_batch, ent_batch)
            else:
                _ = model(x_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)

    batch_ms = np.median(batch_times)
    throughput = batch_size / (batch_ms / 1000)

    return {
        "single_sample_ms": round(float(single_ms), 4),
        "single_p99_ms": round(float(single_p99), 4),
        "batch64_ms": round(float(batch_ms), 4),
        "per_sample_in_batch_ms": round(float(batch_ms / batch_size), 4),
        "throughput_samples_sec": round(float(throughput), 1),
    }


def benchmark_model(
    name: str,
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 100,
    iters: int = 500,
    entropy_shape: tuple = None,
) -> dict:
    """Full benchmark for a model."""
    logger.info("Benchmarking %s on %s…", name, device)

    params = count_parameters(model)
    logger.info(
        "  Parameters: %d (%.2f MB)",
        params["total"], params["size_mb"],
    )

    latency = benchmark_latency(model, input_shape, device,
                                warmup=warmup, iterations=iters,
                                entropy_shape=entropy_shape)
    logger.info(
        "  Single-sample: %.3f ms (p99: %.3f ms)",
        latency["single_sample_ms"], latency["single_p99_ms"],
    )
    logger.info(
        "  Batch-64: %.3f ms (%.3f ms/sample, %.0f samples/s)",
        latency["batch64_ms"],
        latency["per_sample_in_batch_ms"],
        latency["throughput_samples_sec"],
    )

    return {"name": name, "parameters": params, "latency": latency}


def benchmark_onnx(
    onnx_path: str,
    seq_len: int = 100,
    warmup: int = 100,
    iters: int = 500,
) -> dict:
    """Benchmark ONNX Runtime inference (CPU-only)."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX benchmark")
        return {"name": "ONNX BiMamba+Ent", "error": "onnxruntime not installed"}

    logger.info("Benchmarking ONNX BiMamba+Entropy (CPU via ORT)…")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    seq_single = np.random.randn(1, seq_len, 3).astype(np.float32)
    ent_single = np.random.randn(1, 6).astype(np.float32)
    feeds_single = {input_names[0]: seq_single, input_names[1]: ent_single}

    # Warmup
    for _ in range(warmup):
        sess.run(None, feeds_single)

    # Single sample
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds_single)
        times.append((time.perf_counter() - t0) * 1000)

    single_ms = float(np.median(times))
    single_p99 = float(np.percentile(times, 99))

    # Batch 64
    seq_batch = np.random.randn(64, seq_len, 3).astype(np.float32)
    ent_batch = np.random.randn(64, 6).astype(np.float32)
    feeds_batch = {input_names[0]: seq_batch, input_names[1]: ent_batch}

    for _ in range(warmup // 2):
        sess.run(None, feeds_batch)

    batch_times = []
    for _ in range(iters // 2):
        t0 = time.perf_counter()
        sess.run(None, feeds_batch)
        batch_times.append((time.perf_counter() - t0) * 1000)

    batch_ms = float(np.median(batch_times))
    throughput = 64 / (batch_ms / 1000)

    # Model size from file
    file_size_mb = os.path.getsize(onnx_path) / 1e6

    logger.info("  Single-sample: %.3f ms (p99: %.3f ms)",
                single_ms, single_p99)
    logger.info("  Batch-64: %.3f ms (%.0f samples/s)", batch_ms, throughput)
    logger.info("  File size: %.2f MB", file_size_mb)

    return {
        "name": "ONNX BiMamba+Ent (CPU)",
        "parameters": {"total": 177608, "size_mb": round(file_size_mb, 3)},
        "latency": {
            "single_sample_ms": round(single_ms, 4),
            "single_p99_ms": round(single_p99, 4),
            "batch64_ms": round(batch_ms, 4),
            "per_sample_in_batch_ms": round(batch_ms / 64, 4),
            "throughput_samples_sec": round(throughput, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Split-Mamba models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="results/benchmark_results.json")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Device: %s", device)

    seq_input = (args.seq_len, 3)   # Sequence models: (100, 3)
    tab_input = (39,)                # Tabular models: 39 features
    ent_shape = (6,)                 # Entropy features
    results = []

    # ── 1. DataPlane MLP (tabular mode — data plane) ─────────────────
    dp_mlp = DataPlaneCNN(num_classes=args.num_classes,
                          input_features=39)
    results.append(benchmark_model("DataPlaneMLP (tabular)", dp_mlp,
                                   tab_input, device,
                                   warmup=args.warmup, iters=args.iterations))

    # ── 2. SequenceCNN (baseline controller) ──────────────────────────
    seq_cnn = SequenceCNN(num_classes=args.num_classes, seq_len=args.seq_len)
    results.append(benchmark_model("SequenceCNN", seq_cnn, seq_input, device,
                                   warmup=args.warmup, iters=args.iterations))

    # ── 3. BiMamba + Entropy (proposed — control plane) ───────────────
    bimamba_ent = BiMambaController(
        num_classes=args.num_classes,
        d_model=64, n_layers=4,
        d_state=16, d_conv=4, expand=1,
        n_entropy_features=6, use_entropy=True,
        input_dim=3, dropout=0.15,
    )
    results.append(benchmark_model(
        "BiMamba+Entropy (proposed)", bimamba_ent, seq_input, device,
        warmup=args.warmup, iters=args.iterations,
        entropy_shape=ent_shape))

    # ── 4. BiMamba (no entropy — ablation) ────────────────────────────
    bimamba_noent = BiMambaController(
        num_classes=args.num_classes,
        d_model=64, n_layers=4,
        d_state=16, d_conv=4, expand=1,
        n_entropy_features=6, use_entropy=False,
        input_dim=3, dropout=0.15,
    )
    results.append(benchmark_model(
        "BiMamba (no entropy)", bimamba_noent, seq_input, device,
        warmup=args.warmup, iters=args.iterations))

    # ── 5. MambaController (unidirectional — comparison) ──────────────
    mamba_uni = MambaController(
        num_classes=args.num_classes,
        d_model=64, n_layers=4,
        d_state=16, d_conv=4, expand=1,
        sequence_length=args.seq_len, input_dim=3,
    )
    results.append(benchmark_model(
        "Mamba-Uni (d=64, L=4)", mamba_uni, seq_input, device,
        warmup=args.warmup, iters=args.iterations))

    # ── 6. ONNX BiMamba+Entropy ───────────────────────────────────────
    onnx_path = "models/onnx/bimamba_ent.onnx"
    if os.path.exists(onnx_path):
        onnx_result = benchmark_onnx(onnx_path, args.seq_len,
                                     warmup=args.warmup, iters=args.iterations)
        results.append(onnx_result)
    else:
        logger.warning("ONNX model not found at %s — skipping", onnx_path)

    # ── Summary table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("%-30s %10s %10s %12s %12s", "Model",
                "Params", "Size(MB)", "Single(ms)", "Throughput")
    logger.info("-" * 90)
    for r in results:
        params = r.get("parameters", {})
        lat = r.get("latency", {})
        logger.info(
            "%-30s %10s %10s %12s %10s/s",
            r["name"],
            str(params.get("total", "n/a")),
            str(params.get("size_mb", "n/a")),
            str(lat.get("single_sample_ms", "n/a")),
            str(lat.get("throughput_samples_sec", "n/a")),
        )
    logger.info("=" * 90)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to %s", args.output)


if __name__ == "__main__":
    main()
