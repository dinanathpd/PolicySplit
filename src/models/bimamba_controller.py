"""
BiMamba Controller — Bidirectional Selective State Space Model
with Entropy-Augmented Feature Fusion for IoT DDoS Detection.

Architecture:
    ┌──────────────────────────────────────┐
    │  Packet Sequence (B, L, 3)           │
    │    log(pkt_len), log(IAT), direction │
    └──────────────┬───────────────────────┘
                   │
              Input Embed (3 → d_model)
                   │
           ┌───────┴───────┐
           │               │
    Forward Mamba     Backward Mamba
      (n_layers)        (n_layers)
           │               │
           └───────┬───────┘
                   │
             Concat (2·d_model)  ← BiMamba output
                   │
            Projection (2·d_model → d_model)
                   │
           Global Attention Pool
                   │
                h_temporal (d_model)
                   │
    ┌──────────────┴───────────────┐
    │                              │
    │    Entropy Features (B, 6)   │
    │  src_ip_H, dst_ip_H, ...    │
    │         ↓                    │
    │    MLP (6 → d_model//2)      │
    │         = h_spatial          │
    │                              │
    └──────────────┬───────────────┘
                   │
        Concat [h_temporal; h_spatial]
                   │
          Classification Head
            → num_classes logits

Novelty: Bidirectional SSM captures both causal and anti-causal
temporal patterns. Entropy features inject spatial diversity
information that resolves the DDoS/DoS ambiguity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba_controller import SelectiveSSM, MambaBlock, FocalLoss


# ── Attention Pooling ─────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Learned attention pooling over sequence dimension."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L, D) → (B, D)
        mask: (B, L) bool, True = valid position
        """
        scores = self.attn(x).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)  # (B, L)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)


# ── BiMamba Encoder ───────────────────────────────────────────────────

class BiMambaEncoder(nn.Module):
    """
    Bidirectional Mamba: runs forward + backward SSM scans
    and fuses their outputs.

    This captures both causal (past→future) and anti-causal
    (future→past) temporal dependencies — critical for packet
    sequences where attack signatures appear at different positions.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        # Forward scan layers
        self.fwd_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.fwd_norm = nn.LayerNorm(d_model)

        # Backward scan layers (separate parameters)
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])
            self.bwd_norm = nn.LayerNorm(d_model)

        # Fuse forward + backward
        fan_in = 2 * d_model if bidirectional else d_model
        self.fuse_proj = nn.Linear(fan_in, d_model, bias=False)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        # Forward scan
        h = x
        for layer in self.fwd_layers:
            h = layer(h)
        h = self.fwd_norm(h)  # (B, L, D)

        if self.bidirectional:
            # Backward scan: reverse, process, reverse back
            h_bwd = x.flip(dims=[1])
            for layer in self.bwd_layers:
                h_bwd = layer(h_bwd)
            h_bwd = self.bwd_norm(h_bwd).flip(dims=[1])
            h = torch.cat([h, h_bwd], dim=-1)  # (B, L, 2D)

        h = self.fuse_proj(h)  # (B, L, D)
        h = self.fuse_norm(h)
        h = self.dropout(h)

        # Residual
        return h + x


# ── Entropy Feature Encoder ──────────────────────────────────────────

class EntropyEncoder(nn.Module):
    """
    MLP encoder for spatial entropy features.

    Input: (B, n_entropy_features) — typically 6 features
    Output: (B, d_out) — projected to match temporal embedding dim
    """

    def __init__(self, n_features: int = 6, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full BiMamba Controller ──────────────────────────────────────────

class BiMambaController(nn.Module):
    """
    Bidirectional Mamba with entropy-augmented feature fusion.

    Designed for the SDN control plane: processes "ambiguous" flows
    escalated from the data-plane CNN and leverages spatial entropy
    to resolve DDoS/DoS confusion.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 8).
    d_model : int
        Hidden dimension for Mamba layers.
    n_layers : int
        Number of Mamba layers per direction (total = 2 × n_layers).
    d_state : int
        SSM state dimension.
    d_conv : int
        Causal convolution kernel size.
    expand : int
        Expansion factor for inner dimension.
    dropout : float
        Dropout rate.
    input_dim : int
        Per-timestep feature dimension (default 3: pkt_len, IAT, dir).
    n_entropy_features : int
        Number of entropy features (default 6).
    use_entropy : bool
        Whether to use entropy features (set False if not available).
    """

    def __init__(
        self,
        num_classes: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        input_dim: int = 3,
        n_entropy_features: int = 6,
        use_entropy: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_entropy = use_entropy
        self.bidirectional = bidirectional

        # Input embedding: raw packet features → d_model
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(
            # supports up to 512 timesteps
            torch.randn(1, 512, d_model) * 0.02)

        # BiMamba encoder
        self.encoder = BiMambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Attention pooling
        self.pool = AttentionPool(d_model)

        # Entropy encoder (optional)
        d_entropy = d_model // 2 if use_entropy else 0
        if use_entropy:
            self.entropy_encoder = EntropyEncoder(
                n_features=n_entropy_features, d_out=d_entropy)

        # Classification head
        cls_input_dim = d_model + d_entropy
        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_input_dim),
            nn.Dropout(dropout),
            nn.Linear(cls_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        entropy_features: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : (B, L, input_dim)
            Packet-level features.
        entropy_features : (B, n_entropy_features), optional
            Spatial entropy features. Required if use_entropy=True.
        mask : (B, L) bool, optional
            True = valid timestep, False = padding.

        Returns
        -------
        logits : (B, num_classes)
        """
        B, L, _ = sequences.shape

        # Embed + positional encoding
        x = self.input_embedding(sequences)  # (B, L, d_model)
        x = x + self.pos_embed[:, :L, :]

        # BiMamba encode
        h_seq = self.encoder(x)  # (B, L, d_model)

        # Pool over sequence
        h_temporal = self.pool(h_seq, mask)  # (B, d_model)

        # Fuse with entropy features
        if self.use_entropy and entropy_features is not None:
            h_spatial = self.entropy_encoder(
                entropy_features)  # (B, d_model//2)
            h = torch.cat([h_temporal, h_spatial], dim=-1)
        else:
            # Pad to expected classifier input dim if entropy not used at inference
            if self.use_entropy:
                pad = torch.zeros(B, self.d_model // 2,
                                  device=h_temporal.device)
                h = torch.cat([h_temporal, pad], dim=-1)
            else:
                h = h_temporal

        logits = self.classifier(h)  # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
