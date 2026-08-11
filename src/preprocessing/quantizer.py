"""
Feature Quantizer — Convert continuous features to integers for P4 compatibility.

Two quantization stages:
  1. Packet-length bucketing (4-bit, 12 tokens)
  2. IAT normalisation to uint8

These match the P4 register widths specified in the implementation document.
"""

import numpy as np
from typing import Optional


# -----------------------------------------------------------------------
# Packet-length bucketing   (Table 4 in implementation spec)
# -----------------------------------------------------------------------

LENGTH_BUCKET_EDGES = [
    0, 40, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1500
]
# Token IDs: 0–11   (12–15 reserved)
N_LENGTH_TOKENS = 12


def quantize_packet_length(lengths: np.ndarray) -> np.ndarray:
    """
    Bucketise packet lengths into 4-bit tokens (0–11).

    Parameters
    ----------
    lengths : array of float/int  — raw packet lengths in bytes.

    Returns
    -------
    tokens : array of uint8 in [0, 11].
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    tokens = np.digitize(lengths, LENGTH_BUCKET_EDGES) - 1
    tokens = np.clip(tokens, 0, N_LENGTH_TOKENS - 1).astype(np.uint8)
    return tokens


def quantize_packet_length_log(lengths: np.ndarray) -> np.ndarray:
    """
    Alternative: log2 bucketing as in Equation (9) of the spec.

    Q_L(L) = floor(log2(L/40))  if L > 40 else 0,  clamped to [0, 11].
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    tokens = np.where(
        lengths > 40,
        np.floor(np.log2(lengths / 40.0)),
        0.0,
    )
    return np.clip(tokens, 0, 11).astype(np.uint8)


# -----------------------------------------------------------------------
# IAT normalisation   (Equation 10 in implementation spec)
# -----------------------------------------------------------------------

DEFAULT_MEDIAN_IAT_MS = 10.0   # empirical median from training data


def normalise_iat(
    iat_ms: np.ndarray,
    median_iat: float = DEFAULT_MEDIAN_IAT_MS,
) -> np.ndarray:
    """
    Normalise inter-arrival times to uint8 [0, 255].

    Δt_norm = clip( ⌊ log(1 + Δt) / log(1 + μ_IAT) × 255 ⌋, 0, 255 )
    """
    iat_ms = np.asarray(iat_ms, dtype=np.float64)
    denom = np.log(1.0 + median_iat)
    normed = np.floor(np.log(1.0 + iat_ms) / denom * 255.0)
    return np.clip(normed, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------
# Full quantizer
# -----------------------------------------------------------------------

class FeatureQuantizer:
    """
    Quantize raw flow sequences into integer representations.

    A flow is [(pkt_len, iat_ms, direction), …].
    Output is (seq_len, 3) uint8 array.
    """

    def __init__(
        self,
        sequence_length: int = 100,
        median_iat: float = DEFAULT_MEDIAN_IAT_MS,
        length_method: str = "bucket",   # "bucket" or "log"
    ):
        self.sequence_length = sequence_length
        self.median_iat = median_iat
        self.length_method = length_method

    def quantize_flow(self, packets: list) -> np.ndarray:
        """
        Quantize a single flow's packet list.

        Parameters
        ----------
        packets : list of (pkt_len, iat_ms, direction)

        Returns
        -------
        seq : ndarray of shape (sequence_length, 3), dtype uint8
              Columns: [quantized_length, normalised_IAT, direction]
        """
        seq = np.zeros((self.sequence_length, 3), dtype=np.uint8)
        n = min(len(packets), self.sequence_length)

        if n == 0:
            return seq

        raw = np.array(packets[:n], dtype=np.float64)
        lengths = raw[:, 0]
        iats = raw[:, 1]
        dirs = raw[:, 2].astype(np.uint8)

        if self.length_method == "log":
            q_lengths = quantize_packet_length_log(lengths)
        else:
            q_lengths = quantize_packet_length(lengths)

        q_iats = normalise_iat(iats, self.median_iat)

        seq[:n, 0] = q_lengths
        seq[:n, 1] = q_iats
        seq[:n, 2] = dirs

        return seq

    def quantize_batch(self, flows: list) -> np.ndarray:
        """Quantize a list of flows → (N, seq_len, 3) uint8 array."""
        out = np.zeros(
            (len(flows), self.sequence_length, 3), dtype=np.uint8
        )
        for i, pkts in enumerate(flows):
            out[i] = self.quantize_flow(pkts)
        return out

    def compute_median_iat(self, flows: list) -> float:
        """Compute median IAT across all flows (for calibration)."""
        all_iats = []
        for pkts in flows:
            for _, iat, _ in pkts:
                if iat > 0:
                    all_iats.append(iat)
        if all_iats:
            self.median_iat = float(np.median(all_iats))
        return self.median_iat
