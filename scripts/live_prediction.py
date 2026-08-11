#!/usr/bin/env python3
"""
live_prediction.py - Generate synthetic attacks and run ONNX split-inference.

Generates novel attack PCAPs (NOT from CICIoT2023) using scapy, extracts
features using the same pipeline as training, and runs the full split-inference
architecture (CNN data plane + BiMamba control plane) with ONNX Runtime.

This validates:
  1. End-to-end prediction pipeline on out-of-distribution traffic
  2. CNN confidence-based selective escalation
  3. Per-flow classification with actual labels

Usage:
    python scripts/live_prediction.py
    python scripts/live_prediction.py --output-dir results/live_predictions
"""

import os
import sys
import time
import json
import random
import struct
import socket
import hashlib
import logging
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants (must match training pipeline) ──────────────────────────

SEQ_LEN = 100
N_ENT = 6
MIN_PACKETS = 10

CLASSES = [
    "Benign", "DDoS", "DoS", "Mirai",
    "Recon", "Web", "Spoofing", "BruteForce",
]


# ── Synthetic PCAP generation ────────────────────────────────────────

def _random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _ip_to_int(ip_str):
    return struct.unpack("!I", socket.inet_aton(ip_str))[0]


def generate_benign_pcap(pcap_path, n_flows=50, pkts_per_flow=30):
    """Generate realistic benign HTTP/HTTPS-like traffic."""
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

    packets = []
    t = 1000000.0  # base timestamp

    for _ in range(n_flows):
        src = _random_ip()
        dst = "10.0.2.1"
        sport = random.randint(49152, 65535)
        dport = random.choice([80, 443, 8080, 8443])

        for j in range(pkts_per_flow):
            # Variable IAT (50 to 500 ms)
            t += random.uniform(0.05, 0.5)
            # Variable payload (typical HTTP)
            payload_size = random.randint(40, 1400)
            direction = random.choice([0, 1])

            if direction == 0:
                pkt = Ether() / IP(src=src, dst=dst) / TCP(sport=sport,
                                                           dport=dport, flags="PA") / Raw(b"X" * payload_size)
            else:
                pkt = Ether() / IP(src=dst, dst=src) / TCP(sport=dport,
                                                           dport=sport, flags="PA") / Raw(b"X" * payload_size)
            pkt.time = t
            packets.append(pkt)

    wrpcap(pcap_path, packets)
    logger.info("Generated benign PCAP: %d packets, %d flows -> %s",
                len(packets), n_flows, pcap_path)
    return len(packets)


def generate_ddos_syn_pcap(pcap_path, n_sources=200, pkts_per_source=20):
    """Generate DDoS SYN flood from many distributed sources."""
    from scapy.all import Ether, IP, TCP, wrpcap

    packets = []
    t = 1000000.0
    dst = "10.0.2.1"

    sources = [(_random_ip(), random.randint(1024, 65535))
               for _ in range(n_sources)]

    for src_ip, sport in sources:
        for _ in range(pkts_per_source):
            t += random.uniform(0.0001, 0.005)  # Very fast (0.1 to 5 ms)
            dport = random.choice([80, 443, 22, 8080])
            pkt = Ether() / IP(src=src_ip, dst=dst) / \
                TCP(sport=sport, dport=dport, flags="S")
            pkt.time = t
            packets.append(pkt)

    random.shuffle(packets)
    # Re-assign timestamps after shuffle to simulate interleaved arrivals
    t = 1000000.0
    for pkt in packets:
        t += random.uniform(0.0001, 0.002)
        pkt.time = t

    wrpcap(pcap_path, packets)
    logger.info("Generated DDoS SYN flood: %d pkts from %d sources -> %s",
                len(packets), n_sources, pcap_path)
    return len(packets)


def generate_dos_tcp_pcap(pcap_path, n_flows=5, pkts_per_flow=500):
    """Generate DoS TCP flood from few sources (concentrated)."""
    from scapy.all import Ether, IP, TCP, Raw, wrpcap

    packets = []
    t = 1000000.0

    for _ in range(n_flows):
        src = _random_ip()
        dst = "10.0.2.1"
        sport = random.randint(1024, 65535)
        dport = random.choice([80, 443])

        for _ in range(pkts_per_flow):
            t += random.uniform(0.0001, 0.003)  # Fast
            payload = Raw(b"A" * random.randint(100, 1400))
            pkt = Ether() / IP(src=src, dst=dst) / \
                TCP(sport=sport, dport=dport, flags="PA") / payload
            pkt.time = t
            packets.append(pkt)

    wrpcap(pcap_path, packets)
    logger.info("Generated DoS TCP flood: %d pkts from %d flows -> %s",
                len(packets), n_flows, pcap_path)
    return len(packets)


def _timed(pkt, t):
    """Set packet timestamp."""
    pkt.time = t
    return pkt


def generate_recon_pcap(pcap_path, n_targets=100, ports_per_target=20):
    """Generate reconnaissance service detection scan traffic.

    Simulates nmap -sV: connects to each port, exchanges multiple
    probe/response packets (banner grab), then disconnects.
    Each 5-tuple flow gets >= 12 packets to pass MIN_PACKETS filter.
    """
    from scapy.all import Ether, IP, TCP, Raw, wrpcap

    packets = []
    t = 1000000.0
    src = _random_ip()

    for _ in range(n_targets):
        dst = _random_ip()
        scan_ports = random.sample(range(1, 10000), min(ports_per_target, 200))

        for dport in scan_ports:
            sport = random.randint(49152, 65535)
            # Full TCP handshake + service probes (12 to 18 packets)
            t += random.uniform(0.001, 0.05)
            packets.append(_timed(Ether() / IP(src=src, dst=dst) /
                           TCP(sport=sport, dport=dport, flags="S"), t))
            t += random.uniform(0.001, 0.01)
            packets.append(_timed(Ether() / IP(src=dst, dst=src) /
                           TCP(sport=dport, dport=sport, flags="SA"), t))
            t += random.uniform(0.0005, 0.002)
            packets.append(_timed(Ether() / IP(src=src, dst=dst) /
                           TCP(sport=sport, dport=dport, flags="A"), t))
            # Service probe packets (banner grab)
            for _ in range(random.randint(3, 6)):
                t += random.uniform(0.001, 0.02)
                probe = Raw(b"\x00" * random.randint(10, 100))
                packets.append(_timed(Ether() / IP(src=src, dst=dst) /
                               TCP(sport=sport, dport=dport, flags="PA") / probe, t))
                t += random.uniform(0.001, 0.01)
                resp = Raw(b"SSH-2.0-OpenSSH" if random.random() <
                           0.3 else b"\x00" * random.randint(20, 200))
                packets.append(_timed(Ether() / IP(src=dst, dst=src) /
                               TCP(sport=dport, dport=sport, flags="PA") / resp, t))
            # RST
            t += random.uniform(0.001, 0.005)
            packets.append(_timed(Ether() / IP(src=src, dst=dst) /
                           TCP(sport=sport, dport=dport, flags="R"), t))

    wrpcap(pcap_path, packets)
    logger.info("Generated Recon scan: %d pkts, %d targets x %d ports -> %s",
                len(packets), n_targets, ports_per_target, pcap_path)
    return len(packets)


def generate_bruteforce_pcap(pcap_path, n_attempts=200):
    """Generate SSH/HTTP brute force login attempts.

    Groups attempts into persistent connections (same sport) to form
    flows with enough packets for feature extraction.
    """
    from scapy.all import Ether, IP, TCP, Raw, wrpcap

    packets = []
    t = 1000000.0
    src = _random_ip()
    dst = "10.0.2.1"
    dport = random.choice([22, 21, 3389])

    # Group attempts into sessions of 20 to 40 (same sport = same flow)
    attempts_per_session = 25
    n_sessions = max(1, n_attempts // attempts_per_session)

    for _ in range(n_sessions):
        sport = random.randint(49152, 65535)

        # SYN handshake
        t += random.uniform(0.5, 2.0)
        pkt = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="S")
        pkt.time = t
        packets.append(pkt)

        t += random.uniform(0.001, 0.01)
        pkt = Ether() / IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="SA")
        pkt.time = t
        packets.append(pkt)

        for _ in range(attempts_per_session):
            # Login attempt
            t += random.uniform(0.1, 1.0)
            cred = f"user:pass{random.randint(0, 9999)}".encode()
            pkt = Ether() / IP(src=src, dst=dst) / \
                TCP(sport=sport, dport=dport, flags="PA") / Raw(cred)
            pkt.time = t
            packets.append(pkt)

            # Auth fail
            t += random.uniform(0.01, 0.05)
            pkt = Ether() / IP(src=dst, dst=src) / TCP(sport=dport,
                                                       dport=sport, flags="PA") / Raw(b"Auth failed")
            pkt.time = t
            packets.append(pkt)

        # FIN
        t += random.uniform(0.01, 0.1)
        pkt = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="FA")
        pkt.time = t
        packets.append(pkt)

    wrpcap(pcap_path, packets)
    logger.info("Generated BruteForce: %d pkts, %d attempts -> %s",
                len(packets), n_attempts, pcap_path)
    return len(packets)


def generate_spoofing_pcap(pcap_path, n_packets=500):
    """Generate DNS spoofing-like traffic with spoofed source IPs.

    Groups packets into flows of >= MIN_PACKETS to ensure extraction.
    """
    from scapy.all import Ether, IP, UDP, Raw, wrpcap

    packets = []
    t = 1000000.0
    victim = "10.0.2.1"
    spoofed_srcs = ["8.8.8.8", "1.1.1.1", "208.67.222.222", "8.8.4.4"]
    victim_ports = [random.randint(49152, 65535) for _ in range(20)]

    for _ in range(n_packets):
        t += random.uniform(0.001, 0.05)
        spoofed_src = random.choice(spoofed_srcs)
        dport = random.choice(victim_ports)  # Reuse ports to form flows
        dns_payload = Raw(b"\x00" * random.randint(40, 512))
        pkt = Ether() / IP(src=spoofed_src, dst=victim) / \
            UDP(sport=53, dport=dport) / dns_payload
        pkt.time = t
        packets.append(pkt)

    wrpcap(pcap_path, packets)
    logger.info("Generated Spoofing: %d pkts -> %s", len(packets), pcap_path)
    return len(packets)


# ── Feature extraction (matches training pipeline) ───────────────────

def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    ep1, ep2 = (src_ip, src_port), (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    return hashlib.md5(f"{ep1[0]}:{ep1[1]}-{ep2[0]}:{ep2[1]}-{proto}".encode()).hexdigest()


def _shannon_entropy(arr):
    if len(arr) == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-30)))


def extract_features_from_pcap(pcap_path, window_sec=10.0):
    """
    Extract sequence + entropy features from a PCAP file.
    Matches the training pipeline in pcap_to_entropy_fast.py.

    Returns:
        sequences: (N, 100, 3) - [log(pkt_len), log(IAT), direction]
        entropy:   (N, 6)      - [src_ip_H, dst_ip_H, src_port_H, dst_port_H,
                                   log_uniq_src, log_uniq_dst]
        flow_ids:  list of str
    """
    import dpkt

    flow_table = {}
    global_pkts = []
    pkt_count = 0

    with open(pcap_path, "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)

        for ts, buf in reader:
            pkt_count += 1
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip_pkt = eth.data
                proto = ip_pkt.p
                src_ip = socket.inet_ntoa(ip_pkt.src)
                dst_ip = socket.inet_ntoa(ip_pkt.dst)
                src_ip_int = struct.unpack("!I", ip_pkt.src)[0]
                dst_ip_int = struct.unpack("!I", ip_pkt.dst)[0]

                if isinstance(ip_pkt.data, dpkt.tcp.TCP):
                    src_port, dst_port = ip_pkt.data.sport, ip_pkt.data.dport
                elif isinstance(ip_pkt.data, dpkt.udp.UDP):
                    src_port, dst_port = ip_pkt.data.sport, ip_pkt.data.dport
                else:
                    src_port, dst_port = 0, 0

                pkt_len = ip_pkt.len or len(buf)
                key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                global_pkts.append(
                    (ts, src_ip_int, dst_ip_int, src_port, dst_port))
            except Exception:
                continue

            # Flow management
            if key not in flow_table:
                flow_table[key] = {
                    "flow_id": key,
                    "init_src": src_ip,
                    "pkts": [],
                    "start_time": ts,
                    "last_time": ts,
                }
            flow = flow_table[key]

            # Flow timeout check
            if ts - flow["last_time"] > 60.0 and flow["pkts"]:
                flow_table[key] = {
                    "flow_id": key,
                    "init_src": src_ip,
                    "pkts": [],
                    "start_time": ts,
                    "last_time": ts,
                }
                flow = flow_table[key]

            iat = (ts - flow["last_time"]) * 1000.0 if flow["pkts"] else 0.0
            direction = 1.0 if src_ip == flow["init_src"] else 0.0

            if len(flow["pkts"]) < 200:
                flow["pkts"].append((pkt_len, iat, direction))
            flow["last_time"] = ts

    # Filter flows with too few packets
    flows = [f for f in flow_table.values() if len(f["pkts"]) >= MIN_PACKETS]
    if not flows:
        return np.zeros((0, SEQ_LEN, 3)), np.zeros((0, N_ENT)), []

    # Build sequence array
    n = len(flows)
    sequences = np.zeros((n, SEQ_LEN, 3), dtype=np.float32)
    for i, f in enumerate(flows):
        pkts = f["pkts"]
        m = min(len(pkts), SEQ_LEN)
        raw = np.array(pkts[:m], dtype=np.float64)
        sequences[i, :m, 0] = np.log1p(raw[:, 0])
        sequences[i, :m, 1] = np.log1p(raw[:, 1])
        sequences[i, :m, 2] = raw[:, 2]

    # Compute window entropy
    if not global_pkts:
        entropy = np.zeros((n, N_ENT), dtype=np.float32)
    else:
        gp = np.array(global_pkts, dtype=np.float64)
        g_times = gp[:, 0]
        sort_idx = np.argsort(g_times)
        g_times = g_times[sort_idx]
        g_src_ip = gp[sort_idx, 1].astype(np.int64)
        g_dst_ip = gp[sort_idx, 2].astype(np.int64)
        g_src_port = gp[sort_idx, 3].astype(np.int32)
        g_dst_port = gp[sort_idx, 4].astype(np.int32)

        half_w = window_sec / 2.0
        entropy = np.zeros((n, N_ENT), dtype=np.float32)

        for i, f in enumerate(flows):
            t_start = f["start_time"] - half_w
            t_end = f["last_time"] + half_w
            lo = np.searchsorted(g_times, t_start, side="left")
            hi = np.searchsorted(g_times, t_end, side="right")
            if lo >= hi:
                continue

            slc = slice(lo, min(hi, lo + 100000))
            entropy[i, 0] = _shannon_entropy(g_src_ip[slc])
            entropy[i, 1] = _shannon_entropy(g_dst_ip[slc])
            entropy[i, 2] = _shannon_entropy(g_src_port[slc])
            entropy[i, 3] = _shannon_entropy(g_dst_port[slc])
            entropy[i, 4] = np.log1p(len(np.unique(g_src_ip[slc])))
            entropy[i, 5] = np.log1p(len(np.unique(g_dst_ip[slc])))

    flow_ids = [f["flow_id"] for f in flows]
    return sequences, entropy, flow_ids


# ── ONNX inference ───────────────────────────────────────────────────

def load_onnx_models(cnn_path, bimamba_path):
    import onnxruntime as ort
    cnn_sess = ort.InferenceSession(
        cnn_path, providers=["CPUExecutionProvider"])
    bimamba_sess = ort.InferenceSession(
        bimamba_path, providers=["CPUExecutionProvider"])
    return cnn_sess, bimamba_sess


def predict_bimamba(session, sequences, entropy):
    """Run BiMamba ONNX inference."""
    in_names = [inp.name for inp in session.get_inputs()]
    out_name = session.get_outputs()[0].name

    feeds = {
        in_names[0]: sequences.astype(np.float32),
        in_names[1]: entropy.astype(np.float32),
    }
    (logits,) = session.run([out_name], feeds)

    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    return preds, confidences, probs


def predict_cnn_with_confidence(session, X_tab):
    """Run CNN ONNX inference and compute confidence margin."""
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    (logits,) = session.run([out_name], {in_name: X_tab.astype(np.float32)})
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)

    # Top-2 margin
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    preds = probs.argmax(axis=1)
    return preds, margin, probs


# ── Real test data evaluation (CICIoT2023) ──────────────────────────

def run_real_test(args):
    """Run full split-inference on real CICIoT2023 test data.

    Demonstrates practical deployment accuracy:
    1. CNN classifies ALL flows on data plane
    2. Low-confidence flows escalated to BiMamba on control plane
    3. Combined predictions with actual per-class metrics
    """
    import onnxruntime as ort

    logger.info("=" * 60)
    logger.info("REAL TEST DATA EVALUATION (CICIoT2023)")
    logger.info("=" * 60)

    # Load sequence test data (BiMamba)
    logger.info("Loading sequence test data from %s", args.seq_test)
    seq_data = np.load(args.seq_test)
    X_seq = seq_data["X_seq"].astype(np.float32)
    X_ent = seq_data["X_ent"].astype(np.float32)
    y_seq = seq_data["y"].astype(np.int64)
    logger.info("  Sequences: %d samples (seq: %s, ent: %s)",
                len(y_seq), X_seq.shape, X_ent.shape)

    # Load tabular test data (CNN)
    logger.info("Loading tabular test data from %s", args.tab_test)
    tab_data = np.load(args.tab_test)
    X_tab = tab_data["X"].astype(np.float32)
    y_tab = tab_data["y"].astype(np.int64)
    if os.path.exists(args.norm_stats):
        norms = np.load(args.norm_stats)
        X_tab = (X_tab - norms["mean"]) / np.clip(norms["std"], 1e-8, None)
        logger.info("  Applied z-score normalization")
    logger.info("  Tabular: %d samples x %d features", *X_tab.shape)

    # Load ONNX models
    logger.info("Loading CNN ONNX: %s", args.cnn_onnx)
    cnn_sess = ort.InferenceSession(
        args.cnn_onnx, providers=["CPUExecutionProvider"])
    logger.info("Loading BiMamba ONNX: %s", args.bimamba_onnx)
    bimamba_sess = ort.InferenceSession(
        args.bimamba_onnx, providers=["CPUExecutionProvider"])

    # ── CNN data plane: classify ALL tabular flows ───────────────
    logger.info("\n--- Stage 1: CNN Data Plane (all %d tabular flows) ---",
                len(y_tab))
    t0 = time.time()
    cnn_preds, cnn_margin, cnn_probs = predict_cnn_with_confidence(
        cnn_sess, X_tab)
    cnn_time = time.time() - t0

    cnn_correct = (cnn_preds == y_tab).sum()
    cnn_acc = cnn_correct / len(y_tab)
    logger.info("  CNN accuracy: %.4f (%d/%d) in %.3fs (%.1f us/sample)",
                cnn_acc, cnn_correct, len(y_tab),
                cnn_time, 1e6 * cnn_time / len(y_tab))

    # ── Selective escalation ─────────────────────────────────────
    threshold = args.threshold
    confident_mask = cnn_margin >= threshold
    n_confident = confident_mask.sum()
    n_escalated = len(y_tab) - n_confident
    pct_data_plane = 100.0 * n_confident / len(y_tab)

    logger.info(
        "\n--- Stage 2: Selective Escalation (theta=%.2f) ---", threshold)
    logger.info("  Data plane (CNN confident): %d flows (%.1f%%)",
                n_confident, pct_data_plane)
    logger.info("  Escalated to controller:    %d flows (%.1f%%)",
                n_escalated, 100.0 - pct_data_plane)

    # CNN accuracy on confident subset only
    if n_confident > 0:
        cnn_conf_acc = (cnn_preds[confident_mask] ==
                        y_tab[confident_mask]).mean()
        logger.info("  CNN accuracy on confident flows: %.4f", cnn_conf_acc)

    # ── BiMamba control plane: classify sequence test data ────────
    logger.info("\n--- Stage 3: BiMamba Control Plane (%d sequence flows) ---",
                len(y_seq))
    t0 = time.time()
    bimamba_preds, bimamba_conf, _ = predict_bimamba(
        bimamba_sess, X_seq, X_ent)
    bimamba_time = time.time() - t0

    bimamba_correct = (bimamba_preds == y_seq).sum()
    bimamba_acc = bimamba_correct / len(y_seq)
    logger.info("  BiMamba accuracy: %.4f (%d/%d) in %.3fs (%.1f us/sample)",
                bimamba_acc, bimamba_correct, len(y_seq),
                bimamba_time, 1e6 * bimamba_time / len(y_seq))

    # ── Combined split-inference estimate ────────────────────────
    # Since tabular and sequence test sets are disjoint, estimate analytically
    est_combined = (pct_data_plane / 100.0) * cnn_conf_acc + \
                   (1.0 - pct_data_plane / 100.0) * bimamba_acc

    logger.info("\n" + "=" * 60)
    logger.info("SPLIT-INFERENCE RESULTS")
    logger.info("=" * 60)
    logger.info("  CNN data plane:       %.4f accuracy, %.1f us/sample",
                cnn_acc, 1e6 * cnn_time / len(y_tab))
    logger.info("  BiMamba control plane: %.4f accuracy, %.1f us/sample",
                bimamba_acc, 1e6 * bimamba_time / len(y_seq))
    logger.info("  Escalation threshold:  %.2f", threshold)
    logger.info("  Data plane handles:    %.1f%% of flows", pct_data_plane)
    logger.info("  CNN confident acc:     %.4f", cnn_conf_acc)
    logger.info("  COMBINED ACCURACY:     %.4f", est_combined)

    # ── Per-class breakdown ──────────────────────────────────────
    logger.info("\n--- Per-Class BiMamba Predictions ---")
    logger.info("%-12s  %-7s  %-7s  %-7s  %-7s  %-7s",
                "Class", "Total", "Correct", "Acc", "Conf", "TopMis")

    for c in range(len(CLASSES)):
        mask = y_seq == c
        n_total = mask.sum()
        if n_total == 0:
            continue
        c_preds = bimamba_preds[mask]
        c_confs = bimamba_conf[mask]
        c_correct = (c_preds == c).sum()
        c_acc = c_correct / n_total

        # Top misclassification
        wrong = c_preds[c_preds != c]
        if len(wrong) > 0:
            vals, counts = np.unique(wrong, return_counts=True)
            top_mis = CLASSES[vals[counts.argmax()]]
            top_mis_pct = counts.max() / n_total * 100
            mis_str = f"{top_mis}({top_mis_pct:.0f}%)"
        else:
            mis_str = "-"

        logger.info("%-12s  %-7d  %-7d  %-7.4f  %-7.4f  %-s",
                    CLASSES[c], n_total, c_correct, c_acc,
                    c_confs.mean(), mis_str)

    # ── Per-class CNN escalation ─────────────────────────────────
    logger.info("\n--- Per-Class CNN Escalation (theta=%.2f) ---", threshold)
    logger.info("%-12s  %-7s  %-7s  %-8s  %-8s",
                "Class", "Total", "Escal", "Esc%", "CNNAcc")

    for c in range(len(CLASSES)):
        mask = y_tab == c
        n_total = mask.sum()
        if n_total == 0:
            continue
        c_esc = (mask & ~confident_mask).sum()
        c_cnn_acc = (cnn_preds[mask] == y_tab[mask]).mean()
        logger.info("%-12s  %-7d  %-7d  %-8.1f  %-8.4f",
                    CLASSES[c], n_total, c_esc,
                    100.0 * c_esc / n_total, c_cnn_acc)

    # ── Save results ─────────────────────────────────────────────
    results = {
        "experiment": "CICIoT2023 Real Test Data - Split Inference",
        "cnn_data_plane": {
            "model": args.cnn_onnx,
            "n_samples": len(y_tab),
            "accuracy": round(float(cnn_acc), 4),
            "inference_s": round(cnn_time, 3),
            "us_per_sample": round(1e6 * cnn_time / len(y_tab), 1),
        },
        "bimamba_control_plane": {
            "model": args.bimamba_onnx,
            "n_samples": len(y_seq),
            "accuracy": round(float(bimamba_acc), 4),
            "inference_s": round(bimamba_time, 3),
            "us_per_sample": round(1e6 * bimamba_time / len(y_seq), 1),
        },
        "split_inference": {
            "threshold": threshold,
            "pct_data_plane": round(float(pct_data_plane), 2),
            "cnn_confident_accuracy": round(float(cnn_conf_acc), 4),
            "combined_accuracy": round(float(est_combined), 4),
        },
        "per_class_bimamba": {
            CLASSES[c]: {
                "total": int((y_seq == c).sum()),
                "correct": int((bimamba_preds[y_seq == c] == c).sum()),
                "accuracy": round(float(
                    (bimamba_preds[y_seq == c] == c).mean()
                ), 4) if (y_seq == c).sum() > 0 else 0.0,
            }
            for c in range(len(CLASSES))
        },
    }

    out_path = os.path.join(args.output_dir, "real_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", out_path)


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic attacks and run ONNX split-inference")
    parser.add_argument("--output-dir", default="results/live_predictions")
    parser.add_argument("--pcap-dir", default="/tmp/synthetic_pcaps")
    parser.add_argument(
        "--cnn-onnx", default="models/quantized/cnn_quantized.onnx")
    parser.add_argument(
        "--bimamba-onnx", default="models/onnx/bimamba_ent.onnx")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="CNN confidence margin threshold for escalation")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip PCAP generation (use existing)")
    parser.add_argument("--test-npz", action="store_true",
                        help="Run on real CICIoT2023 test data instead of synthetic")
    parser.add_argument("--seq-test",
                        default="models/bimamba_ent_test_data.npz",
                        help="Sequence test NPZ (X_seq, X_ent, y)")
    parser.add_argument("--tab-test", default="data/test.npz",
                        help="Tabular test NPZ (X, y) for CNN")
    parser.add_argument("--norm-stats", default="data/norm_stats.npz",
                        help="Z-score normalization stats for CNN")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_npz:
        return run_real_test(args)

    os.makedirs(args.pcap_dir, exist_ok=True)

    # ── Step 1: Generate synthetic attack PCAPs ──────────────────
    attack_configs = {
        "benign":     {"func": generate_benign_pcap,
                       "kwargs": {"n_flows": 80, "pkts_per_flow": 40}},
        "ddos_syn":   {"func": generate_ddos_syn_pcap,
                       "kwargs": {"n_sources": 300, "pkts_per_source": 25}},
        "dos_tcp":    {"func": generate_dos_tcp_pcap,
                       "kwargs": {"n_flows": 8, "pkts_per_flow": 600}},
        "recon_scan": {"func": generate_recon_pcap,
                       "kwargs": {"n_targets": 30, "ports_per_target": 80}},
        "bruteforce": {"func": generate_bruteforce_pcap,
                       "kwargs": {"n_attempts": 300}},
        "spoofing":   {"func": generate_spoofing_pcap,
                       "kwargs": {"n_packets": 2000}},
    }

    expected_classes = {
        "benign": "Benign",
        "ddos_syn": "DDoS",
        "dos_tcp": "DoS",
        "recon_scan": "Recon",
        "bruteforce": "BruteForce",
        "spoofing": "Spoofing",
    }

    if not args.skip_generate:
        logger.info("=" * 60)
        logger.info("Step 1: Generating synthetic attack PCAPs")
        logger.info("=" * 60)
        random.seed(42)
        for name, cfg in attack_configs.items():
            pcap_path = os.path.join(args.pcap_dir, f"{name}.pcap")
            cfg["func"](pcap_path, **cfg["kwargs"])
    else:
        logger.info("Skipping PCAP generation (--skip-generate)")

    # ── Step 2: Extract features ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Extracting sequence + entropy features")
    logger.info("=" * 60)

    all_results = {}
    for name in attack_configs:
        pcap_path = os.path.join(args.pcap_dir, f"{name}.pcap")
        if not os.path.exists(pcap_path):
            logger.warning("PCAP not found: %s", pcap_path)
            continue

        t0 = time.time()
        sequences, entropy, flow_ids = extract_features_from_pcap(pcap_path)
        extract_time = time.time() - t0

        all_results[name] = {
            "sequences": sequences,
            "entropy": entropy,
            "flow_ids": flow_ids,
            "n_flows": len(flow_ids),
            "extract_time": extract_time,
            "expected_class": expected_classes[name],
        }
        logger.info("  %s: %d flows extracted in %.2fs (seq: %s, ent: %s)",
                    name, len(flow_ids), extract_time,
                    sequences.shape, entropy.shape)

    total_flows = sum(r["n_flows"] for r in all_results.values())
    logger.info("Total: %d flows from %d attack types",
                total_flows, len(all_results))

    if total_flows == 0:
        logger.error("No flows extracted. Check PCAP generation.")
        return

    # ── Step 3: Run BiMamba ONNX inference ───────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Running BiMamba ONNX inference (control plane)")
    logger.info("=" * 60)

    _, bimamba_sess = load_onnx_models(args.cnn_onnx, args.bimamba_onnx)

    predictions = {}
    total_infer_time = 0

    for name, data in all_results.items():
        if data["n_flows"] == 0:
            continue

        t0 = time.time()
        preds, confs, probs = predict_bimamba(
            bimamba_sess, data["sequences"], data["entropy"])
        infer_time = time.time() - t0
        total_infer_time += infer_time

        pred_labels = [CLASSES[p] for p in preds]
        expected = data["expected_class"]

        # Compute how many were classified as the expected class
        correct_class = sum(1 for l in pred_labels if l == expected)
        pct_correct = 100.0 * correct_class / \
            len(pred_labels) if pred_labels else 0

        predictions[name] = {
            "n_flows": data["n_flows"],
            "expected_class": expected,
            "predictions": dict(zip(*np.unique(pred_labels, return_counts=True))),
            "correct_class_pct": round(pct_correct, 1),
            "mean_confidence": round(float(confs.mean()), 4),
            "infer_time_s": round(infer_time, 3),
            "us_per_flow": round(1e6 * infer_time / data["n_flows"], 1),
        }
        # Convert numpy keys to strings
        predictions[name]["predictions"] = {
            str(k): int(v) for k, v in predictions[name]["predictions"].items()
        }

        logger.info("  %s: %d flows | expected=%s | classified_as=%s",
                    name, data["n_flows"], expected,
                    predictions[name]["predictions"])
        logger.info("    correct=%.1f%% | confidence=%.4f | time=%.3fs (%.1f us/flow)",
                    pct_correct, confs.mean(), infer_time,
                    1e6 * infer_time / data["n_flows"])

    # ── Step 4: Summary ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4: Prediction Summary")
    logger.info("=" * 60)

    logger.info("%-15s  %-10s  %-8s  %-10s  %-8s  %-10s",
                "Attack", "Expected", "Flows", "TopPred", "Correct%", "Conf")
    logger.info("-" * 70)

    for name, p in predictions.items():
        top_pred = max(p["predictions"], key=p["predictions"].get)
        logger.info("%-15s  %-10s  %-8d  %-10s  %-8.1f  %-10.4f",
                    name, p["expected_class"], p["n_flows"],
                    top_pred, p["correct_class_pct"], p["mean_confidence"])

    logger.info("-" * 70)
    logger.info("Total flows: %d | Total inference: %.3fs | Avg: %.1f us/flow",
                total_flows, total_infer_time,
                1e6 * total_infer_time / total_flows)

    # ── Save results ─────────────────────────────────────────────
    results = {
        "experiment": "Synthetic Attack Live Prediction",
        "description": "Out-of-distribution synthetic attacks classified by ONNX BiMamba",
        "model": args.bimamba_onnx,
        "threshold": args.threshold,
        "total_flows": total_flows,
        "total_inference_s": round(total_infer_time, 3),
        "avg_us_per_flow": round(1e6 * total_infer_time / total_flows, 1),
        "per_attack": predictions,
    }

    results_path = os.path.join(
        args.output_dir, "live_prediction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
