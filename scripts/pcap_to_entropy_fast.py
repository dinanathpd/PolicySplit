#!/usr/bin/env python3
"""
pcap_to_entropy_fast.py - Parallel PCAP extraction with dpkt + multiprocessing.

10-50x faster than the scapy-based sequential version.

Produces HDF5 with:
    /sequences  (N, 100, 3)  - [log(pkt_len), log(IAT), direction]
    /entropy    (N, 6)       - [src_ip_H, dst_ip_H, src_port_H, dst_port_H,
                                 log_uniq_src, log_uniq_dst]
    /labels     (N,)         - class label
    /flow_ids   (N,)         - MD5 flow key

Usage:
    PYTHONPATH=. python scripts/pcap_to_entropy_fast.py \
        --pcap-dir dataset_cic_iot_2023/PCAP \
        --output data/sequences/entropy_flows_v2.h5 \
        --workers 10 --window 10.0
"""

from src.preprocessing.feature_extractor import (
    ATTACK_LABEL_TO_GROUP,
    GROUP_LABEL_TO_ID,
    ID_TO_GROUP_LABEL,
)
import os
import sys
import math
import hashlib
import struct
import socket
import logging
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import dpkt
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

N_ENT = 6
SEQ_LEN = 100
MIN_PACKETS = 10
FLOW_TIMEOUT = 60.0
MAX_PKTS_PER_FLOW = 200
MAX_PKTS_TOTAL = 1_500_000
MAX_FLOWS_PER_DIR = 30_000

DIR_TO_LABEL = {
    "Backdoor_Malware": "BACKDOOR_MALWARE",
    "Benign_Final": "BENIGN",
    "BrowserHijacking": "BROWSERHIJACKING",
    "CommandInjection": "COMMANDINJECTION",
    "DDoS-ACK_Fragmentation": "DDOS-ACK_FRAGMENTATION",
    "DDoS-HTTP_Flood": "DDOS-HTTP_FLOOD",
    "DDoS-ICMP_Flood": "DDOS-ICMP_FLOOD",
    "DDoS-ICMP_Fragmentation": "DDOS-ICMP_FRAGMENTATION",
    "DDoS-PSHACK_Flood": "DDOS-PSHACK_FLOOD",
    "DDoS-RSTFINFlood": "DDOS-RSTFINFLOOD",
    "DDoS-SlowLoris": "DDOS-SLOWLORIS",
    "DDoS-SYN_Flood": "DDOS-SYN_FLOOD",
    "DDoS-SynonymousIP_Flood": "DDOS-SYNONYMOUSIP_FLOOD",
    "DDoS-TCP_Flood": "DDOS-TCP_FLOOD",
    "DDoS-UDP_Flood": "DDOS-UDP_FLOOD",
    "DDoS-UDP_Fragmentation": "DDOS-UDP_FRAGMENTATION",
    "DictionaryBruteForce": "DICTIONARYBRUTEFORCE",
    "DNS_Spoofing": "DNS_SPOOFING",
    "DoS-HTTP_Flood": "DOS-HTTP_FLOOD",
    "DoS-SYN_Flood": "DOS-SYN_FLOOD",
    "DoS-TCP_Flood": "DOS-TCP_FLOOD",
    "DoS-UDP_Flood": "DOS-UDP_FLOOD",
    "Mirai-greeth_flood": "MIRAI-GREETH_FLOOD",
    "Mirai-greip_flood": "MIRAI-GREIP_FLOOD",
    "Mirai-udpplain": "MIRAI-UDPPLAIN",
    "MITM-ArpSpoofing": "MITM-ARPSPOOFING",
    "Recon-HostDiscovery": "RECON-HOSTDISCOVERY",
    "Recon-OSScan": "RECON-OSSCAN",
    "Recon-PingSweep": "RECON-PINGSWEEP",
    "Recon-PortScan": "RECON-PORTSCAN",
    "SqlInjection": "SQLINJECTION",
    "Uploading_Attack": "UPLOADING_ATTACK",
    "VulnerabilityScan": "VULNERABILITYSCAN",
    "XSS": "XSS",
}


# ── Fast PCAP parsing with dpkt ──────────────────────────────────────

def _inet_ntoa(addr_bytes):
    """Convert 4-byte IP to string."""
    return socket.inet_ntoa(addr_bytes)


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    ep1, ep2 = (src_ip, src_port), (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    return hashlib.md5(f"{ep1[0]}:{ep1[1]}-{ep2[0]}:{ep2[1]}-{proto}".encode()).hexdigest()


def parse_pcap_dpkt(pcap_path: str) -> Tuple[List[Dict], List[Tuple]]:
    """
    Parse PCAP with dpkt (much faster than scapy).

    Returns (flows_list, global_packets_list)
    where global_packets = [(time, src_ip_int, dst_ip_int, src_port, dst_port), ...]
    Using int IPs for speed (avoid string allocation).
    """
    flow_table = {}
    finished = []
    global_pkts = []
    pkt_count = 0
    err_count = 0

    try:
        with open(pcap_path, 'rb') as fh:
            try:
                reader = dpkt.pcap.Reader(fh)
            except ValueError:
                try:
                    fh.seek(0)
                    reader = dpkt.pcapng.Reader(fh)
                except Exception:
                    logger.warning("Cannot parse %s", pcap_path)
                    return [], []

            for ts, buf in reader:
                pkt_count += 1
                if pkt_count > MAX_PKTS_TOTAL:
                    break
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    proto = ip.p

                    # Fast IP conversion to int (for hashing speed)
                    src_ip_b = ip.src
                    dst_ip_b = ip.dst
                    src_ip_str = _inet_ntoa(src_ip_b)
                    dst_ip_str = _inet_ntoa(dst_ip_b)
                    src_ip_int = struct.unpack('!I', src_ip_b)[0]
                    dst_ip_int = struct.unpack('!I', dst_ip_b)[0]

                    if isinstance(ip.data, dpkt.tcp.TCP):
                        src_port = ip.data.sport
                        dst_port = ip.data.dport
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        src_port = ip.data.sport
                        dst_port = ip.data.dport
                    else:
                        src_port = 0
                        dst_port = 0

                    pkt_len = ip.len or len(buf)
                    key = _flow_key(src_ip_str, dst_ip_str,
                                    src_port, dst_port, proto)

                    # Global packet record (ints for speed)
                    global_pkts.append(
                        (ts, src_ip_int, dst_ip_int, src_port, dst_port))

                except Exception:
                    err_count += 1
                    continue

                # Flow table management
                if key in flow_table:
                    flow = flow_table[key]
                    if ts - flow["last_time"] > FLOW_TIMEOUT:
                        finished.append(_finalize(flow))
                        flow_table[key] = _new_flow(
                            key, src_ip_str, dst_ip_str, src_port, dst_port, proto, ts)
                        flow = flow_table[key]
                else:
                    flow_table[key] = _new_flow(
                        key, src_ip_str, dst_ip_str, src_port, dst_port, proto, ts)
                    flow = flow_table[key]

                iat = (ts - flow["last_time"]) * \
                    1000.0 if flow["pkts"] else 0.0
                direction = 1 if src_ip_str == flow["init_src"] else 0

                if len(flow["pkts"]) < MAX_PKTS_PER_FLOW:
                    flow["pkts"].append((pkt_len, iat, direction))

                flow["last_time"] = ts

    except Exception as e:
        logger.warning("Error reading %s: %s", pcap_path, e)

    # Flush remaining
    for flow in flow_table.values():
        if flow["pkts"]:
            finished.append(_finalize(flow))

    logger.info("  [dpkt] %s: %d pkts -> %d flows (errs=%d)",
                os.path.basename(pcap_path), pkt_count, len(finished), err_count)
    return finished, global_pkts


def _new_flow(key, src_ip, dst_ip, src_port, dst_port, proto, ts):
    return {
        "flow_id": key,
        "src_ip": src_ip, "dst_ip": dst_ip,
        "src_port": src_port, "dst_port": dst_port,
        "protocol": proto,
        "pkts": [],
        "start_time": ts, "last_time": ts,
        "init_src": src_ip,
    }


def _finalize(flow):
    return {
        "flow_id": flow["flow_id"],
        "packets": flow["pkts"],
        "start_time": flow["start_time"],
        "end_time": flow["last_time"],
    }


# ── Vectorized entropy ───────────────────────────────────────────────

def _shannon_entropy_fast(arr):
    """Shannon entropy of integer array using numpy."""
    if len(arr) == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-30)))


def compute_window_entropy_vectorized(
    flows: List[Dict],
    global_pkts: List[Tuple],
    window_sec: float,
) -> np.ndarray:
    """
    Vectorized window entropy using numpy arrays + searchsorted.

    Much faster than the pure-Python bisect version.
    """
    n_flows = len(flows)
    result = np.zeros((n_flows, N_ENT), dtype=np.float32)

    if not global_pkts:
        return result

    # Convert to numpy arrays for searchsorted
    gp = np.array(global_pkts, dtype=np.float64)  # (M, 5)
    g_times = gp[:, 0]
    g_src_ip = gp[:, 1].astype(np.int64)
    g_dst_ip = gp[:, 2].astype(np.int64)
    g_src_port = gp[:, 3].astype(np.int32)
    g_dst_port = gp[:, 4].astype(np.int32)

    # Sort by time (should already be, but ensure)
    sort_idx = np.argsort(g_times)
    g_times = g_times[sort_idx]
    g_src_ip = g_src_ip[sort_idx]
    g_dst_ip = g_dst_ip[sort_idx]
    g_src_port = g_src_port[sort_idx]
    g_dst_port = g_dst_port[sort_idx]

    half_w = window_sec / 2.0

    for i, flow in enumerate(flows):
        t_start = flow["start_time"] - half_w
        t_end = flow["end_time"] + half_w

        lo = np.searchsorted(g_times, t_start, side='left')
        hi = np.searchsorted(g_times, t_end, side='right')

        if lo >= hi:
            continue

        # Cap window to 100k packets to avoid memory blow-up
        if hi - lo > 100000:
            step = (hi - lo) // 100000
            slc = slice(lo, hi, step)
        else:
            slc = slice(lo, hi)

        w_src_ip = g_src_ip[slc]
        w_dst_ip = g_dst_ip[slc]
        w_src_p = g_src_port[slc]
        w_dst_p = g_dst_port[slc]

        result[i, 0] = _shannon_entropy_fast(w_src_ip)
        result[i, 1] = _shannon_entropy_fast(w_dst_ip)
        result[i, 2] = _shannon_entropy_fast(w_src_p)
        result[i, 3] = _shannon_entropy_fast(w_dst_p)
        result[i, 4] = np.log1p(len(np.unique(w_src_ip)))
        result[i, 5] = np.log1p(len(np.unique(w_dst_ip)))

    return result


# ── Per-file worker ──────────────────────────────────────────────────

def _worker_parse_file(args):
    """
    Worker function for multiprocessing.Pool.

    Parses one PCAP file and returns (flows, global_pkts, dir_name).
    """
    pcap_path, dir_name = args
    try:
        flows, gpkts = parse_pcap_dpkt(pcap_path)
        return flows, gpkts, dir_name
    except Exception as e:
        logger.warning("Worker error on %s: %s", pcap_path, e)
        return [], [], dir_name


def _worker_process_dir(args):
    """
    Worker: parse all PCAPs in one directory, compute entropy, return arrays.

    This keeps global_packets per-directory (correct for window entropy,
    since entropy should capture concurrent traffic from the same attack scenario).
    """
    dir_path, dir_name, attack_label, class_id, window_sec = args
    t0 = time.time()

    # Collect all PCAP files in dir
    pcap_files = sorted(
        str(p) for p in Path(dir_path).iterdir()
        if p.suffix.lower() in ('.pcap', '.pcapng')
    )
    if not pcap_files:
        return None

    all_flows = []
    all_gpkts = []

    for pf in pcap_files:
        flows, gpkts = parse_pcap_dpkt(pf)
        all_flows.extend(flows)
        all_gpkts.extend(gpkts)
        del flows, gpkts  # free immediately

    if not all_flows:
        return None

    # Cap global packets to prevent OOM during entropy (keep random subset)
    MAX_GPKTS = 2_000_000
    if len(all_gpkts) > MAX_GPKTS:
        logger.info("  [%s] Capping global_pkts %d -> %d",
                    dir_name, len(all_gpkts), MAX_GPKTS)
        all_gpkts.sort(key=lambda x: x[0])
        # Keep evenly spaced subset to preserve temporal coverage
        step = len(all_gpkts) // MAX_GPKTS
        all_gpkts = all_gpkts[::step][:MAX_GPKTS]
    else:
        # Sort global packets by time
        all_gpkts.sort(key=lambda x: x[0])

    # Filter short flows
    before = len(all_flows)
    all_flows = [f for f in all_flows if len(f["packets"]) >= MIN_PACKETS]
    # Cap flows per directory to prevent OOM
    if len(all_flows) > MAX_FLOWS_PER_DIR:
        np.random.shuffle(all_flows)
        all_flows = all_flows[:MAX_FLOWS_PER_DIR]
    logger.info("  [%s] %d -> %d flows (filtered min_pkts=%d, cap=%d), %d global pkts",
                dir_name, before, len(all_flows), MIN_PACKETS, MAX_FLOWS_PER_DIR, len(all_gpkts))

    if not all_flows:
        return None

    # Build sequences array
    n = len(all_flows)
    sequences = np.zeros((n, SEQ_LEN, 3), dtype=np.float32)
    for i, f in enumerate(all_flows):
        pkts = f["packets"]
        m = min(len(pkts), SEQ_LEN)
        raw = np.array(pkts[:m], dtype=np.float64)
        sequences[i, :m, 0] = np.log1p(raw[:, 0])
        sequences[i, :m, 1] = np.log1p(raw[:, 1])
        sequences[i, :m, 2] = raw[:, 2]

    # Compute window entropy (vectorized)
    entropy = compute_window_entropy_vectorized(
        all_flows, all_gpkts, window_sec)

    labels = np.full(n, class_id, dtype=np.int64)
    flow_ids = [f["flow_id"] for f in all_flows]

    elapsed = time.time() - t0
    logger.info("  [%s] DONE: %d flows, entropy %s, %.1fs",
                dir_name, n, entropy.shape, elapsed)

    return {
        "sequences": sequences,
        "entropy": entropy,
        "labels": labels,
        "flow_ids": flow_ids,
        "dir_name": dir_name,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel PCAP extraction with dpkt")
    parser.add_argument("--pcap-dir", default="dataset_cic_iot_2023/PCAP")
    parser.add_argument(
        "--output", default="data/sequences/entropy_flows_v2.h5")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    parser.add_argument("--window", type=float, default=10.0,
                        help="Entropy window in seconds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 3)

    pcap_root = Path(args.pcap_dir)
    if not pcap_root.exists():
        logger.error("PCAP dir not found: %s", pcap_root)
        sys.exit(1)

    # Build task list: (dir_path, dir_name, attack_label, class_id, window)
    tasks = []
    for d in sorted(pcap_root.iterdir()):
        if not d.is_dir() or d.name not in DIR_TO_LABEL:
            continue
        pcaps = [f for f in d.iterdir() if f.suffix.lower()
                 in ('.pcap', '.pcapng')]
        if not pcaps:
            continue

        attack_label = DIR_TO_LABEL[d.name]
        if attack_label not in ATTACK_LABEL_TO_GROUP:
            continue
        group = ATTACK_LABEL_TO_GROUP[attack_label]
        class_id = GROUP_LABEL_TO_ID[group]

        tasks.append((str(d), d.name, attack_label, class_id, args.window))

    logger.info("Found %d directories to process with %d workers",
                len(tasks), n_workers)

    t_start = time.time()

    # Parallel processing
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker_process_dir, tasks)
    else:
        results = [_worker_process_dir(t) for t in tasks]

    # Collect results
    all_seq, all_ent, all_lab, all_fids = [], [], [], []
    for r in results:
        if r is None:
            continue
        all_seq.append(r["sequences"])
        all_ent.append(r["entropy"])
        all_lab.append(r["labels"])
        all_fids.extend(r["flow_ids"])
        logger.info("  Collected %s: %d flows",
                    r["dir_name"], len(r["labels"]))

    if not all_seq:
        logger.error("No flows extracted!")
        sys.exit(1)

    X_seq = np.concatenate(all_seq)
    X_ent = np.concatenate(all_ent)
    y = np.concatenate(all_lab)

    elapsed = time.time() - t_start
    logger.info("Total: %d flows extracted in %.1fs", len(y), elapsed)
    logger.info("  Sequences: %s", X_seq.shape)
    logger.info("  Entropy:   %s", X_ent.shape)

    # Print entropy stats per class
    ent_names = ["src_ip_H", "dst_ip_H", "src_port_H", "dst_port_H",
                 "log_uniq_src", "log_uniq_dst"]
    for cls in np.unique(y):
        mask = y == cls
        name = ID_TO_GROUP_LABEL.get(int(cls), str(cls))
        means = X_ent[mask].mean(axis=0)
        stds = X_ent[mask].std(axis=0)
        logger.info("  Class %d (%s): %d flows", cls, name, mask.sum())
        for j, en in enumerate(ent_names):
            logger.info("    %s: mean=%.3f std=%.3f", en, means[j], stds[j])

    # Save HDF5
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as hf:
        hf.create_dataset("sequences", data=X_seq,
                          compression="gzip", chunks=True)
        hf.create_dataset("entropy", data=X_ent,
                          compression="gzip", chunks=True)
        hf.create_dataset("labels", data=y)
        dt = h5py.string_dtype()
        hf.create_dataset("flow_ids", data=all_fids, dtype=dt)
        meta = hf.create_group("metadata")
        meta.attrs["seq_len"] = SEQ_LEN
        meta.attrs["n_classes"] = 8
        meta.attrs["total_flows"] = len(y)
        meta.attrs["n_entropy_features"] = N_ENT
        meta.attrs["window_seconds"] = args.window

    logger.info("Saved to %s (%.1f MB)",
                args.output, os.path.getsize(args.output) / 1e6)


if __name__ == "__main__":
    main()
