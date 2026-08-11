"""
mamba_inference.py — Mamba model inference engine for the SDN controller.
"""

from src.models.mamba_controller import MambaController
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MambaInferenceEngine:
    """
    Wraps the trained Mamba model for real-time inference
    on telemetry data received from the P4 switch.
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int = 8,
        d_model: int = 256,
        n_layers: int = 6,
        sequence_length: int = 100,
        device: str = "cuda",
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.sequence_length = sequence_length

        self.model = MambaController(
            num_classes=num_classes,
            d_model=d_model,
            n_layers=n_layers,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, telemetry_data: dict) -> dict:
        """
        Run Mamba inference on a telemetry payload.

        Parameters
        ----------
        telemetry_data : dict from telemetry_parser.parse_telemetry_packet()

        Returns
        -------
        dict with keys: class, confidence, probabilities
        """
        seq_len = telemetry_data["seq_len"]
        pkt_lengths = telemetry_data["packet_lengths"]
        iat_values = telemetry_data["iat_values"]

        # Build feature matrix (seq_length, 3)
        sequence = np.zeros((self.sequence_length, 3), dtype=np.float32)
        for i in range(min(seq_len, self.sequence_length)):
            sequence[i, 0] = pkt_lengths[i] if i < len(pkt_lengths) else 0
            sequence[i, 1] = iat_values[i] if i < len(iat_values) else 0
            # Default direction (not available in telemetry)
            sequence[i, 2] = 1

        seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return {
            "class": pred_class,
            "confidence": confidence,
            "probabilities": probs.cpu().numpy()[0].tolist(),
        }
