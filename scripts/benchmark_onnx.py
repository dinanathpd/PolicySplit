#!/usr/bin/env python3
"""
benchmark_onnx.py - Lightweight latency and throughput benchmark using ONNX Runtime.

No PyTorch required. Measures the exact numbers needed for the paper:
  - Single-sample inference latency (median, P99)
  - Batch inference latency and throughput
  - Telemetry parse time (Scapy)
  - End-to-end controller pipeline timing

Usage:
    python scripts/benchmark_onnx.py --model models/onnx/bimamba_ent.onnx
    python scripts/benchmark_onnx.py --model models/onnx/bimamba_ent.onnx --full-pipeline
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller.onnx_inference import OnnxInferenceEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def benchmark_single_sample(engine, warmup=50, iterations=500):
    """Measure single-sample inference latency."""
    seq = np.random.randn(1, 100, 3).astype(np.float32)
    ent = np.random.randn(1, 6).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        engine.predict_batch(seq, ent)

    # Timed runs
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.predict_batch(seq, ent)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = sorted(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(times[0]),
        "max_ms": float(times[-1]),
        "iterations": iterations,
    }


def benchmark_batch(engine, batch_size=64, warmup=20, iterations=200):
    """Measure batch inference latency and throughput."""
    seq = np.random.randn(batch_size, 100, 3).astype(np.float32)
    ent = np.random.randn(batch_size, 6).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        engine.predict_batch(seq, ent)

    # Timed runs
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.predict_batch(seq, ent)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = sorted(times)
    median_ms = float(np.median(times))
    throughput = batch_size / (median_ms / 1000) if median_ms > 0 else 0

    return {
        "batch_size": batch_size,
        "median_ms": median_ms,
        "mean_ms": float(np.mean(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "per_sample_ms": median_ms / batch_size,
        "throughput_samples_sec": round(throughput, 1),
        "iterations": iterations,
    }


def benchmark_telemetry_parse(iterations=1000):
    """Measure telemetry packet parsing time (Scapy)."""
    try:
        from scapy.all import IP, UDP, Raw
        from controller.telemetry_parser import parse_telemetry_packet, TelemetryHeader
    except ImportError:
        logger.warning(
            "Scapy not available, skipping telemetry parse benchmark")
        return None

    if TelemetryHeader is None:
        logger.warning("TelemetryHeader not defined, skipping")
        return None

    # Build a synthetic telemetry packet
    telem_payload = TelemetryHeader(
        flow_id=12345,
        seq_len=16,
        cnn_pred_class=1,
        cnn_confidence=750,
        pkt_lengths=0x123456789ABCDEF0,
        iat_values=0xAABBCCDDEEFF0011,
    )
    pkt = IP(dst="10.0.3.1") / UDP(dport=9999) / telem_payload

    # Warmup
    for _ in range(50):
        parse_telemetry_packet(pkt)

    # Timed
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        parse_telemetry_packet(pkt)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times = sorted(times)
    return {
        "median_us": float(np.median(times)),
        "mean_us": float(np.mean(times)),
        "p99_us": float(np.percentile(times, 99)),
        "iterations": iterations,
    }


def benchmark_full_pipeline(engine, iterations=200):
    """
    Measure end-to-end controller pipeline:
    telemetry parse + inference + decision.
    """
    try:
        from scapy.all import IP, UDP
        from controller.telemetry_parser import parse_telemetry_packet, TelemetryHeader
    except ImportError:
        logger.warning("Scapy not available, skipping pipeline benchmark")
        return None

    if TelemetryHeader is None:
        return None

    telem_payload = TelemetryHeader(
        flow_id=12345, seq_len=16, cnn_pred_class=1,
        cnn_confidence=750,
        pkt_lengths=0x123456789ABCDEF0,
        iat_values=0xAABBCCDDEEFF0011,
    )
    pkt = IP(dst="10.0.3.1") / UDP(dport=9999) / telem_payload

    # Warmup
    for _ in range(30):
        data = parse_telemetry_packet(pkt)
        engine.predict(data)

    # Timed
    times_parse = []
    times_infer = []
    times_total = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        data = parse_telemetry_packet(pkt)
        t1 = time.perf_counter()
        result = engine.predict(data)
        t2 = time.perf_counter()

        times_parse.append((t1 - t0) * 1e6)
        times_infer.append((t2 - t1) * 1000)
        times_total.append((t2 - t0) * 1000)

    return {
        "parse_median_us": float(np.median(times_parse)),
        "parse_p99_us": float(np.percentile(times_parse, 99)),
        "inference_median_ms": float(np.median(times_infer)),
        "inference_p99_ms": float(np.percentile(times_infer, 99)),
        "total_median_ms": float(np.median(times_total)),
        "total_p99_ms": float(np.percentile(times_total, 99)),
        "iterations": iterations,
    }


def get_model_info(model_path):
    """Get ONNX model file size."""
    size_bytes = os.path.getsize(model_path)
    return {
        "path": model_path,
        "size_kb": round(size_bytes / 1024, 1),
        "size_mb": round(size_bytes / (1024 * 1024), 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX inference for Split-Mamba"
    )
    parser.add_argument(
        "--model", type=str, default="models/onnx/bimamba_ent.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for batch latency test",
    )
    parser.add_argument(
        "--iterations", type=int, default=500,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Also benchmark full telemetry parse + inference pipeline",
    )
    parser.add_argument(
        "--output", type=str, default="results/benchmark_onnx.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    results = {}

    # Model info
    logger.info("Model: %s", args.model)
    results["model"] = get_model_info(args.model)

    # Load engine
    engine = OnnxInferenceEngine(args.model)

    # Single sample
    logger.info("Benchmarking single-sample latency (%d iters)...",
                args.iterations)
    results["single_sample"] = benchmark_single_sample(
        engine, iterations=args.iterations
    )
    logger.info(
        "  Single: median=%.3f ms, P99=%.3f ms",
        results["single_sample"]["median_ms"],
        results["single_sample"]["p99_ms"],
    )

    # Batch
    logger.info("Benchmarking batch latency (BS=%d, %d iters)...",
                args.batch_size, args.iterations)
    results["batch"] = benchmark_batch(
        engine, batch_size=args.batch_size, iterations=args.iterations
    )
    logger.info(
        "  Batch: median=%.3f ms, throughput=%.0f samples/sec",
        results["batch"]["median_ms"],
        results["batch"]["throughput_samples_sec"],
    )

    # Telemetry parse
    logger.info("Benchmarking telemetry parsing...")
    telem = benchmark_telemetry_parse()
    if telem:
        results["telemetry_parse"] = telem
        logger.info(
            "  Parse: median=%.1f us, P99=%.1f us",
            telem["median_us"], telem["p99_us"],
        )

    # Full pipeline
    if args.full_pipeline:
        logger.info("Benchmarking full pipeline (parse + infer)...")
        pipeline = benchmark_full_pipeline(engine)
        if pipeline:
            results["pipeline"] = pipeline
            logger.info(
                "  Pipeline: total median=%.3f ms (parse=%.1f us + infer=%.3f ms)",
                pipeline["total_median_ms"],
                pipeline["parse_median_us"],
                pipeline["inference_median_ms"],
            )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Summary table for paper
    print("\n" + "=" * 55)
    print("  LATENCY SUMMARY (for paper Table 8)")
    print("=" * 55)
    ss = results["single_sample"]
    bs = results["batch"]
    print(
        f"  BiMamba single-sample:  {ss['median_ms']:.3f} ms (P99: {ss['p99_ms']:.3f} ms)")
    print(
        f"  BiMamba batch (BS={bs['batch_size']}):  {bs['median_ms']:.3f} ms total")
    print(f"  BiMamba per-sample:     {bs['per_sample_ms']:.3f} ms")
    print(
        f"  Throughput:             {bs['throughput_samples_sec']:.0f} samples/sec")
    if telem:
        print(
            f"  Telemetry parse:        {telem['median_us']:.1f} us (P99: {telem['p99_us']:.1f} us)")
    if "pipeline" in results:
        p = results["pipeline"]
        print(
            f"  E2E escalated flow:     {p['total_median_ms']:.3f} ms (P99: {p['total_p99_ms']:.3f} ms)")
    print("=" * 55)


if __name__ == "__main__":
    main()
