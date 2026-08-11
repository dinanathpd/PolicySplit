"""
onnx_inference.py - Lightweight ONNX Runtime inference engine for the SDN controller.

Replaces torch-based MambaInferenceEngine for deployment on machines
without PyTorch installed. Uses only numpy + onnxruntime (~50MB total).

Usage:
    engine = OnnxInferenceEngine("models/onnx/bimamba_ent.onnx")
    result = engine.predict(telemetry_data)
"""

import logging
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)

ID_TO_LABEL = {
    0: "Benign",
    1: "DDoS",
    2: "DoS",
    3: "Mirai",
    4: "Recon",
    5: "Web",
    6: "Spoofing",
    7: "BruteForce",
}


class OnnxInferenceEngine:
    """
    ONNX Runtime inference engine for the BiMamba+Entropy model.

    Inputs:
        sequences:        float32 [1, 100, 3]  (pkt_len, iat, direction)
        entropy_features: float32 [1, 6]       (4 entropy + 2 log_uniq)
    Output:
        logits:           float32 [1, 8]
    """

    def __init__(
        self,
        model_path: str,
        sequence_length: int = 100,
        n_entropy_features: int = 6,
    ):
        if ort is None:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        self.sequence_length = sequence_length
        self.n_entropy_features = n_entropy_features

        # Prefer CPU for deterministic latency measurement
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Cache input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(
            "ONNX model loaded: %s (inputs=%s, outputs=%s)",
            model_path, self.input_names, self.output_names,
        )

    def predict(self, telemetry_data: dict) -> dict:
        """
        Run inference on a telemetry payload from the P4 switch.

        Parameters
        ----------
        telemetry_data : dict
            From telemetry_parser.parse_telemetry_packet(). Keys:
            flow_id, seq_len, cnn_prediction, confidence,
            packet_lengths (list[int]), iat_values (list[int]).

        Returns
        -------
        dict with keys: class, label, confidence, probabilities
        """
        seq_len = telemetry_data["seq_len"]
        pkt_lengths = telemetry_data["packet_lengths"]
        iat_values = telemetry_data["iat_values"]

        # Build sequence tensor [1, 100, 3]
        sequence = np.zeros(
            (1, self.sequence_length, 3), dtype=np.float32
        )
        for i in range(min(seq_len, self.sequence_length)):
            if i < len(pkt_lengths):
                sequence[0, i, 0] = pkt_lengths[i]
            if i < len(iat_values):
                sequence[0, i, 1] = iat_values[i]
            sequence[0, i, 2] = 1.0  # direction placeholder

        # Entropy features [1, 6] - zeros when not available from telemetry
        # In deployment, entropy is computed from accumulated flow state
        entropy = np.zeros(
            (1, self.n_entropy_features), dtype=np.float32
        )

        # Run inference
        feeds = {
            self.input_names[0]: sequence,
            self.input_names[1]: entropy,
        }
        (logits,) = self.session.run(self.output_names, feeds)

        # Softmax
        exp_logits = np.exp(logits[0] - logits[0].max())
        probs = exp_logits / exp_logits.sum()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            "class": pred_class,
            "label": ID_TO_LABEL.get(pred_class, str(pred_class)),
            "confidence": confidence,
            "probabilities": probs.tolist(),
        }

    def predict_batch(self, sequences: np.ndarray, entropy: np.ndarray) -> np.ndarray:
        """
        Batch inference for benchmarking.

        Parameters
        ----------
        sequences : float32 array [N, 100, 3]
        entropy   : float32 array [N, 6]

        Returns
        -------
        logits : float32 array [N, 8]
        """
        feeds = {
            self.input_names[0]: sequences.astype(np.float32),
            self.input_names[1]: entropy.astype(np.float32),
        }
        (logits,) = self.session.run(self.output_names, feeds)
        return logits
