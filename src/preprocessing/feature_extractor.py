"""
Feature Extractor — Load CIC-IoT-2023 CSV data, apply label grouping,
and prepare sequences for model training.

Supports two processing modes:
  1. CSV mode  — use the pre-extracted features from the merged CSVs
  2. PCAP mode — use raw packet sequences (via PcapParser)

For CSV mode the 46 statistical features from the dataset are used.
For sequence models the per-packet (length, IAT, direction) is needed.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Label mapping — 34-class → 8-class grouping
# -----------------------------------------------------------------------

# Fine-grained attack labels as they appear in the CSV "Label" column
# (CIC-IoT-2023 merged CSVs use ALL-UPPERCASE labels)
ATTACK_LABEL_TO_GROUP: Dict[str, str] = {
    # DDoS (12)
    "DDOS-SYN_FLOOD":          "DDoS",
    "DDOS-UDP_FLOOD":          "DDoS",
    "DDOS-TCP_FLOOD":          "DDoS",
    "DDOS-ICMP_FLOOD":         "DDoS",
    "DDOS-HTTP_FLOOD":         "DDoS",
    "DDOS-SLOWLORIS":          "DDoS",
    "DDOS-ACK_FRAGMENTATION":  "DDoS",
    "DDOS-UDP_FRAGMENTATION":  "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-PSHACK_FLOOD":       "DDoS",
    "DDOS-RSTFINFLOOD":        "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    # DoS (4)
    "DOS-SYN_FLOOD":           "DoS",
    "DOS-UDP_FLOOD":           "DoS",
    "DOS-TCP_FLOOD":           "DoS",
    "DOS-HTTP_FLOOD":          "DoS",
    # Mirai (3)
    "MIRAI-GREETH_FLOOD":      "Mirai",
    "MIRAI-GREIP_FLOOD":       "Mirai",
    "MIRAI-UDPPLAIN":          "Mirai",
    # Reconnaissance (5)
    "RECON-PINGSWEEP":         "Recon",
    "RECON-OSSCAN":            "Recon",
    "RECON-PORTSCAN":          "Recon",
    "RECON-HOSTDISCOVERY":     "Recon",
    "VULNERABILITYSCAN":       "Recon",
    # Web attacks (6)
    "XSS":                     "Web",
    "SQLINJECTION":            "Web",
    "COMMANDINJECTION":        "Web",
    "UPLOADING_ATTACK":        "Web",
    "BROWSERHIJACKING":        "Web",
    "BACKDOOR_MALWARE":        "Web",
    # Spoofing (2)
    "DNS_SPOOFING":            "Spoofing",
    "MITM-ARPSPOOFING":        "Spoofing",
    # Brute force (1)
    "DICTIONARYBRUTEFORCE":    "BruteForce",
    # Benign (1)
    "BENIGN":                  "Benign",
}

GROUP_LABEL_TO_ID: Dict[str, int] = {
    "Benign":     0,
    "DDoS":       1,
    "DoS":        2,
    "Mirai":      3,
    "Recon":      4,
    "Web":        5,
    "Spoofing":   6,
    "BruteForce": 7,
}

ID_TO_GROUP_LABEL: Dict[int, str] = {
    v: k for k, v in GROUP_LABEL_TO_ID.items()}

# CSV feature columns — 39 features in CIC-IoT-2023 merged CSVs
FEATURE_COLUMNS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP",
    "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Variance",
]

LABEL_COLUMN = "Label"


class FeatureExtractor:
    """
    Load & transform the CIC-IoT-2023 CSV dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to MERGED_CSV directory (contains Merged01.csv … Merged46.csv).
    num_classes : int
        8 for grouped classification, 34 for fine-grained.
    """

    def __init__(self, dataset_dir: str, num_classes: int = 8):
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes

        if num_classes == 8:
            self.label_map = {
                raw: GROUP_LABEL_TO_ID[group]
                for raw, group in ATTACK_LABEL_TO_GROUP.items()
            }
        else:
            # 34-class: alphabetical ordering
            unique_labels = sorted(ATTACK_LABEL_TO_GROUP.keys())
            self.label_map = {lbl: idx for idx,
                              lbl in enumerate(unique_labels)}

    # ------------------------------------------------------------------
    # CSV discovery
    # ------------------------------------------------------------------
    def discover_csvs(self) -> List[Path]:
        """Return sorted list of merged CSV file paths."""
        csvs = sorted(self.dataset_dir.glob("Merged*.csv"))
        if not csvs:
            # Try nested CSV/MERGED_CSV structure
            csvs = sorted(self.dataset_dir.glob("**/MERGED_CSV/Merged*.csv"))
        logger.info("Found %d CSV files in %s", len(csvs), self.dataset_dir)
        return csvs

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_csv(self, csv_path: str, sample_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single merged CSV and return feature matrix + labels.

        When sample_frac < 1.0, uses chunked reading to avoid loading
        the entire file into memory — critical for large CSVs on low-RAM
        systems.

        Returns (X, y) where X is float32 and y is int64.
        """
        if sample_frac < 1.0:
            # Memory-efficient: read in chunks, sample each chunk
            chunks_X, chunks_y = [], []
            rng = np.random.RandomState(42)
            for chunk in pd.read_csv(csv_path, chunksize=50_000):
                chunk = chunk[chunk[LABEL_COLUMN].isin(self.label_map)]
                if len(chunk) == 0:
                    continue
                sampled = chunk.sample(
                    frac=sample_frac, random_state=rng.randint(2**31))
                if len(sampled) == 0:
                    continue
                X_c = sampled[FEATURE_COLUMNS].values.astype(np.float32)
                y_c = sampled[LABEL_COLUMN].map(
                    self.label_map).values.astype(np.int64)
                chunks_X.append(X_c)
                chunks_y.append(y_c)
            if not chunks_X:
                return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), \
                    np.empty((0,), dtype=np.int64)
            X = np.concatenate(chunks_X, axis=0)
            y = np.concatenate(chunks_y, axis=0)
        else:
            df = pd.read_csv(csv_path)
            df = df[df[LABEL_COLUMN].isin(self.label_map)]
            X = df[FEATURE_COLUMNS].values.astype(np.float32)
            y = df[LABEL_COLUMN].map(self.label_map).values.astype(np.int64)

        # Replace NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def load_all(
        self,
        max_files: Optional[int] = None,
        sample_frac: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all merged CSVs and concatenate."""
        csvs = self.discover_csvs()
        if max_files:
            csvs = csvs[:max_files]

        Xs, ys = [], []
        it = tqdm(csvs, desc="Loading CSVs", disable=not verbose)
        for csv_path in it:
            X, y = self.load_csv(str(csv_path), sample_frac=sample_frac)
            Xs.append(X)
            ys.append(y)
            it.set_postfix(samples=sum(len(a) for a in ys))

        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        logger.info("Total samples: %d, features: %d",
                    X_all.shape[0], X_all.shape[1])
        return X_all, y_all

    # ------------------------------------------------------------------
    # Train / Val / Test split (stratified, flow-level)
    # ------------------------------------------------------------------
    @staticmethod
    def stratified_split(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Stratified split into train / val / test sets.

        Returns dict with keys "train", "val", "test",
        each mapping to (X, y) tuple.
        """
        rng = np.random.RandomState(seed)
        classes = np.unique(y)

        train_idx, val_idx, test_idx = [], [], []

        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train: n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

        def _gather(indices):
            indices = np.array(indices)
            return X[indices], y[indices]

        return {
            "train": _gather(train_idx),
            "val": _gather(val_idx),
            "test": _gather(test_idx),
        }

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------
    @staticmethod
    def class_distribution(y: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> np.ndarray:
        """Inverse-frequency class weights: w_c = N_total / (N_c × C)."""
        classes = np.unique(y)
        C = len(classes)
        N = len(y)
        weights = np.zeros(C, dtype=np.float32)
        for c in classes:
            n_c = (y == c).sum()
            weights[c] = N / (n_c * C)
        return weights
