"""
Entropy Feature Computation for Flow-Level Spatial Features.

Computes per-flow entropy features that capture spatial diversity —
the key signal differentiating DDoS (many sources) from DoS (one source).

Features computed per flow:
  1. src_ip_entropy   — Shannon entropy of source IPs in the flow's time window
  2. dst_ip_entropy   — Shannon entropy of destination IPs  
  3. src_port_entropy  — Shannon entropy of source ports
  4. dst_port_entropy  — Shannon entropy of destination ports
  5. unique_src_ips    — Count of unique source IPs (log-scaled)
  6. unique_dst_ips    — Count of unique destination IPs (log-scaled)

For a DoS attack from one IP:  src_ip_entropy ≈ 0, unique_src_ips = 1
For a DDoS attack from 1000 IPs: src_ip_entropy >> 0, unique_src_ips >> 1
"""

import math
import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_ENTROPY_FEATURES = 6  # Number of entropy features per flow


def shannon_entropy(values: List) -> float:
    """Compute Shannon entropy of a list of discrete values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_flow_entropy(flow: Dict) -> np.ndarray:
    """
    Compute entropy features for a single flow dict (INTRA-FLOW only).

    NOTE: This only captures diversity WITHIN a single flow (max 2 IPs).
    For spatial diversity across concurrent flows, use compute_batch_window_entropy().

    Returns: np.ndarray of shape (6,) - entropy feature vector.
    """
    meta = flow.get("packets_meta", [])

    if not meta:
        return np.zeros(N_ENTROPY_FEATURES, dtype=np.float32)

    src_ips = [m[0] for m in meta]
    dst_ips = [m[1] for m in meta]
    src_ports = [m[2] for m in meta]
    dst_ports = [m[3] for m in meta]

    return np.array([
        shannon_entropy(src_ips),
        shannon_entropy(dst_ips),
        shannon_entropy(src_ports),
        shannon_entropy(dst_ports),
        np.log1p(len(set(src_ips))),
        np.log1p(len(set(dst_ips))),
    ], dtype=np.float32)


def compute_batch_window_entropy(
    flows: List[Dict],
    global_packets: List[Tuple],
    window_seconds: float = 10.0,
) -> np.ndarray:
    """
    Compute WINDOW-LEVEL entropy for each flow using ALL concurrent packets.

    This is the correct entropy computation for DDoS/DoS discrimination:
    - DoS flow's window contains packets from ~1 source IP -> low entropy
    - DDoS flow's window contains packets from many source IPs -> high entropy

    Parameters
    ----------
    flows : list of flow dicts with 'start_time' and 'end_time'
    global_packets : sorted list of (time, src_ip, dst_ip, src_port, dst_port)
    window_seconds : time window to extend around each flow's active period

    Returns
    -------
    np.ndarray of shape (len(flows), 6)
    """
    import bisect

    if not global_packets:
        return np.zeros((len(flows), N_ENTROPY_FEATURES), dtype=np.float32)

    # Extract sorted timestamps for binary search
    times = [p[0] for p in global_packets]
    n_total = len(times)

    result = np.zeros((len(flows), N_ENTROPY_FEATURES), dtype=np.float32)

    for i, flow in enumerate(flows):
        t_start = flow.get("start_time", 0.0) - window_seconds / 2
        t_end = flow.get("end_time", flow.get(
            "start_time", 0.0)) + window_seconds / 2

        # Binary search for window boundaries
        idx_lo = bisect.bisect_left(times, t_start)
        idx_hi = bisect.bisect_right(times, t_end)

        if idx_lo >= idx_hi:
            continue

        # Collect all IPs/ports in window (cap at 50k packets for memory)
        window_size = min(idx_hi - idx_lo, 50000)
        if idx_hi - idx_lo > 50000:
            # Sample uniformly
            step = (idx_hi - idx_lo) // window_size
            indices = range(idx_lo, idx_hi, step)
        else:
            indices = range(idx_lo, idx_hi)

        src_ips = []
        dst_ips = []
        src_ports = []
        dst_ports = []
        for j in indices:
            _, sip, dip, sp, dp = global_packets[j]
            src_ips.append(sip)
            dst_ips.append(dip)
            src_ports.append(sp)
            dst_ports.append(dp)

        result[i] = np.array([
            shannon_entropy(src_ips),
            shannon_entropy(dst_ips),
            shannon_entropy(src_ports),
            shannon_entropy(dst_ports),
            np.log1p(len(set(src_ips))),
            np.log1p(len(set(dst_ips))),
        ], dtype=np.float32)

    return result


def compute_window_entropy(
    all_packets: List[Tuple],
    flow_key: str,
    window_seconds: float = 10.0,
    flow_start_time: float = 0.0,
    flow_end_time: float = 0.0,
) -> np.ndarray:
    """
    Compute entropy over a time window around a flow.

    This captures the network-level spatial diversity during the
    flow's active period — not just within the flow itself.

    Parameters
    ----------
    all_packets : list of (timestamp, src_ip, dst_ip, src_port, dst_port, flow_key)
        Global packet list (sorted by time).
    flow_key : str
        The flow to compute entropy for.
    window_seconds : float
        Time window to consider around the flow.
    flow_start_time : float
        Start time of the flow.
    flow_end_time : float
        End time of the flow.

    Returns: np.ndarray of shape (6,)
    """
    t_start = flow_start_time - window_seconds / 2
    t_end = flow_end_time + window_seconds / 2

    src_ips, dst_ips, src_ports, dst_ports = [], [], [], []

    for pkt in all_packets:
        t, sip, dip, sp, dp, _ = pkt
        if t < t_start:
            continue
        if t > t_end:
            break
        src_ips.append(sip)
        dst_ips.append(dip)
        src_ports.append(sp)
        dst_ports.append(dp)

    return np.array([
        shannon_entropy(src_ips),
        shannon_entropy(dst_ips),
        shannon_entropy(src_ports),
        shannon_entropy(dst_ports),
        np.log1p(len(set(src_ips))),
        np.log1p(len(set(dst_ips))),
    ], dtype=np.float32)
