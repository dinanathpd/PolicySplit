"""
PCAP Parser — Extract per-flow packet sequences from PCAP files.

Implements Algorithm 1 from the implementation spec:
    For each packet, extract 5-tuple → group into flows → produce
    per-flow sequences of (packet_length, IAT, direction).
"""

import os
import hashlib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 5-tuple flow key helper
_PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto) -> str:
    """Canonical bidirectional flow key (sorted endpoints)."""
    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    raw = f"{ep1[0]}:{ep1[1]}-{ep2[0]}:{ep2[1]}-{proto}"
    return hashlib.md5(raw.encode()).hexdigest()


class PcapParser:
    """
    Parse PCAP files and extract per-flow packet sequences.

    Parameters
    ----------
    flow_timeout : float
        Seconds of inactivity after which a flow is considered finished.
    max_packets_per_flow : int
        Maximum number of packets to keep for each flow.
    """

    def __init__(self, flow_timeout: float = 60.0, max_packets_per_flow: int = 200,
                 max_packets_total: int = 5_000_000):
        self.flow_timeout = flow_timeout
        self.max_packets = max_packets_per_flow
        self.max_packets_total = max_packets_total

    def parse_file(self, pcap_path: str) -> List[Dict]:
        """
        Parse a single PCAP file using streaming (constant memory).

        Returns
        -------
        flows : list of dict
            Each dict has keys:
                flow_id     : str (MD5 of 5-tuple)
                src_ip      : str
                dst_ip      : str
                src_port    : int
                dst_port    : int
                protocol    : int
                packets     : list of (pkt_len, iat_ms, direction)
                start_time  : float (epoch seconds)
                end_time    : float
        """
        try:
            from scapy.all import PcapReader, IP, TCP, UDP
        except ImportError:
            raise ImportError(
                "scapy is required for PCAP parsing.  pip install scapy"
            )

        logger.info("Streaming %s", pcap_path)

        flow_table: Dict[str, Dict] = {}
        finished_flows: List[Dict] = []
        # (time, src_ip, dst_ip, src_port, dst_port)
        global_packets: List[Tuple] = []
        pkt_count = 0

        err_count = 0
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                try:
                    pkt_count += 1
                    if pkt_count > self.max_packets_total:
                        logger.info("  Hit max_packets_total=%d, stopping",
                                    self.max_packets_total)
                        break
                    if pkt_count % 500_000 == 0:
                        logger.info("  %d packets processed, %d flows so far",
                                    pkt_count, len(flow_table) + len(finished_flows))
                    if not pkt.haslayer(IP):
                        continue

                    ip = pkt[IP]
                    proto = ip.proto
                    src_ip = ip.src
                    dst_ip = ip.dst

                    # Extract ports
                    if pkt.haslayer(TCP):
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    elif pkt.haslayer(UDP):
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                    else:
                        src_port = 0
                        dst_port = 0

                    key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)
                    pkt_time = float(pkt.time)
                    pkt_len = len(pkt)

                    # Accumulate for window-level entropy
                    global_packets.append(
                        (pkt_time, src_ip, dst_ip, src_port, dst_port))
                except Exception as e:
                    err_count += 1
                    if err_count <= 5:
                        logger.debug(
                            "  Skipping bad packet #%d: %s", pkt_count, e)
                    continue

                # --- create or update flow ---
                if key not in flow_table:
                    flow_table[key] = {
                        "flow_id": key,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "protocol": proto,
                        "packets": [],
                        # (src_ip, dst_ip, src_port, dst_port) per pkt
                        "packets_meta": [],
                        "start_time": pkt_time,
                        "last_time": pkt_time,
                        "_init_src": src_ip,  # for direction
                    }

                flow = flow_table[key]

                # Check timeout
                if pkt_time - flow["last_time"] > self.flow_timeout:
                    finished_flows.append(self._finalise(flow))
                    # Start fresh flow
                    flow_table[key] = {
                        "flow_id": key,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "protocol": proto,
                        "packets": [],
                        "packets_meta": [],
                        "start_time": pkt_time,
                        "last_time": pkt_time,
                        "_init_src": src_ip,
                    }
                    flow = flow_table[key]

                # IAT (0 for first packet)
                iat = (pkt_time - flow["last_time"]) * \
                    1000.0 if flow["packets"] else 0.0

                # Direction: 1 = same direction as first packet, 0 = reverse
                direction = 1 if src_ip == flow["_init_src"] else 0

                if len(flow["packets"]) < self.max_packets:
                    flow["packets"].append((pkt_len, iat, direction))
                    flow["packets_meta"].append(
                        (src_ip, dst_ip, src_port, dst_port))

                flow["last_time"] = pkt_time

        if err_count:
            logger.warning("  Skipped %d bad packets in %s",
                           err_count, pcap_path)

        # Export remaining flows
        for flow in flow_table.values():
            if flow["packets"]:
                finished_flows.append(self._finalise(flow))

        logger.info("Extracted %d flows from %s",
                    len(finished_flows), pcap_path)
        return finished_flows, global_packets

    @staticmethod
    def _finalise(flow: Dict) -> Dict:
        """Clean up internal keys before returning a flow."""
        out = {k: v for k, v in flow.items() if not k.startswith("_")}
        out["end_time"] = flow["last_time"]
        return out

    def parse_directory(self, pcap_dir: str) -> Tuple[List[Dict], List[Tuple]]:
        """Parse all PCAP files in a directory. Returns (flows, global_packets)."""
        all_flows = []
        all_global_packets = []
        for fname in sorted(os.listdir(pcap_dir)):
            if fname.lower().endswith((".pcap", ".pcapng")):
                try:
                    flows, gpkts = self.parse_file(
                        os.path.join(pcap_dir, fname))
                    all_flows.extend(flows)
                    all_global_packets.extend(gpkts)
                except Exception as e:
                    logger.warning("  Failed to parse %s: %s — skipping file",
                                   fname, e)
        # Sort global packets by time for efficient window lookups
        all_global_packets.sort(key=lambda x: x[0])
        return all_flows, all_global_packets
