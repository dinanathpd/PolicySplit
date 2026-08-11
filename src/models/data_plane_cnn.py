"""
Data Plane CNN — Lightweight model for P4 switch deployment.

Supports two modes:
  1. Sequence mode (PCAP data): 1D-CNN over (batch, seq_len, 3) packet features
  2. Tabular mode (CSV data):   MLP over (batch, n_features) statistical features

Architecture — Sequence mode (from implementation spec):
    Input  -> (batch, seq_len=100, features=3)
    Conv1D -> 16 filters, kernel=5, stride=1
    ReLU + MaxPool(2)
    Conv1D -> 32 filters, kernel=3, stride=1
    ReLU + MaxPool(2)
    Flatten → Dense(64) → ReLU → Dropout(0.3) → Dense(num_classes)

Architecture — Tabular mode:
    Input  -> (batch, n_features=39)
    Dense(512) → BatchNorm → ReLU → Dropout(0.2)
    Dense(256) → BatchNorm → ReLU → Dropout(0.2)
    Dense(128) → BatchNorm → ReLU → Dropout(0.1)
    Dense(num_classes)
"""

import torch
import torch.nn as nn


class DataPlaneCNN(nn.Module):
    """Lightweight classifier designed for quantization and P4 deployment."""

    def __init__(
        self,
        num_classes: int = 8,
        sequence_length: int = 100,
        input_features: int = None,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes.
        sequence_length : int
            Used only in sequence mode (PCAP data).
        input_features : int or None
            If set, use tabular/MLP mode with this many input features.
            If None, use sequence/CNN mode.
        """
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.tabular_mode = input_features is not None

        if self.tabular_mode:
            self._build_tabular(input_features, num_classes)
        else:
            self._build_cnn(sequence_length, num_classes)

        self._init_weights()

    def _build_tabular(self, input_features: int, num_classes: int):
        """MLP for flat/tabular CSV features."""
        self.net = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def _build_cnn(self, sequence_length: int, num_classes: int):
        """1D-CNN for per-packet sequence data."""
        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self._flat_size = self._compute_flat_size(sequence_length)

        self.fc1 = nn.Linear(self._flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def _compute_flat_size(self, seq_len: int) -> int:
        L = seq_len
        L = (L - 5 + 1)   # conv1
        L = L // 2         # pool1
        L = (L - 3 + 1)   # conv2
        L = L // 2         # pool2
        return L * 32

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Tabular mode:  (batch, n_features)
            Sequence mode: (batch, seq_len, 3)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        if self.tabular_mode:
            return self.net(x)

        # --- Sequence (CNN) mode ---
        x = x.transpose(1, 2)                   # (B, 3, L)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc2(x)

    def predict_with_confidence(self, x: torch.Tensor):
        """Return predicted class, confidence margin, entropy, and probabilities.

        The entropy H(p) = -sum(p_i * log(p_i)) measures overall uncertainty.
        High entropy → ambiguous → escalate to Mamba controller.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)

            top2 = torch.topk(probs, k=2, dim=1)
            p_top1 = top2.values[:, 0]
            p_top2 = top2.values[:, 1]
            margin = p_top1 - p_top2

            # Predictive entropy: H(p) = -sum(p_i * log(p_i))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)

            pred_class = top2.indices[:, 0]

        return pred_class, margin, probs, entropy
