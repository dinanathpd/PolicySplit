"""
Mamba Controller — State-Space Model for control-plane DDoS classification.

Uses Mamba (Selective State Space Model) for long-sequence modelling
with linear-time complexity O(L).

Architecture:
    Input embedding:  input_dim → d_model (256)
    Mamba layers:     n_layers × MambaBlock(d_model, d_state, d_conv, expand)
    Classification:   LayerNorm → GlobalAvgPool → Dropout → Linear(num_classes)

For CSV tabular data, each feature becomes a timestep with input_dim=1,
so the sequence length equals the number of features (e.g. 39).

Trained on "ambiguous" samples that the data-plane CNN cannot classify
with high confidence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Standalone Mamba block (no external dependency on mamba_ssm)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Simplified selective state-space model (Mamba-style)."""

    # Recompute the scan in the backward pass instead of storing its
    # O(log L) intermediates. Disabled automatically under torch.compile,
    # where the elementwise chain is fused and the saving does not pay for
    # the extra forward.
    checkpoint_scan = True

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection → 2 × d_inner (for gating)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters — input-dependent B, C, Δ
        self.x_proj = nn.Linear(
            self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Learnable log(A) — structured diagonal
        A = torch.arange(
            1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, d_model)
        returns : (batch, seq_len, d_model)
        """
        B, L, _ = x.shape

        # --- input projection + gate split ---
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # --- causal conv ---
        x_branch = x_branch.transpose(1, 2)           # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :L]    # causal trim
        x_branch = x_branch.transpose(1, 2)           # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        # --- selective SSM ---
        proj = self.x_proj(x_branch)                   # (B, L, 2*d_state+1)
        B_input = proj[:, :, :self.d_state]
        C_input = proj[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(proj[:, :, -1])                # (B, L)

        # Discretise A
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)

        # Simple scan (sequential — fine for moderate lengths)
        y = self._selective_scan(x_branch, dt, A, B_input, C_input)

        # D skip + SiLU gate
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B_input, C_input):
        """
        Exact parallel selective scan.

        The recurrence h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t is a first
        order linear scan, so it is evaluated with the associative operator

            (a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2)

        applied by Hillis-Steele doubling in ceil(log2 L) steps. Every decay
        factor stays in (0, 1], so unlike the log-space quotient form used
        previously there is no division by an underflowing cumulative
        product and no clamping of the decay exponent.

        The whole scan is gradient-checkpointed: the backward pass recomputes
        it instead of storing the O(log L) intermediate (B, L, D, N) tensors.
        """
        if self.checkpoint_scan and self.training and torch.is_grad_enabled():
            return checkpoint(self._scan_impl, x, dt, A, B_input, C_input,
                              use_reentrant=False)
        return self._scan_impl(x, dt, A, B_input, C_input)

    def _scan_impl(self, x, dt, A, B_input, C_input):
        L = x.shape[1]
        # Decays are exponentials, so the scan always runs in float32.
        with torch.amp.autocast("cuda", enabled=False):
            dt = dt.float()
            x = x.float()
            A = A.float()

            # a: (B, L, D, N) multiplicative decay, in (0, 1] since A < 0.
            a = torch.exp(dt.unsqueeze(-1).unsqueeze(-1) * A)
            # b: (B, L, D, N) additive input term dt_t * x_t * B_t.
            b = (dt.unsqueeze(-1) * x).unsqueeze(-1) * \
                B_input.float().unsqueeze(2)

            step = 1
            while step < L:
                a_prev = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
                b_prev = F.pad(b[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
                b = a * b_prev + b
                a = a * a_prev
                step *= 2

            # b now holds h_t; contract the state dimension with C_t.
            return (b * C_input.float().unsqueeze(2)).sum(dim=-1)


class MambaBlock(nn.Module):
    """Single Mamba layer with pre-norm and residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Full Mamba Controller
# ---------------------------------------------------------------------------

class MambaController(nn.Module):
    """
    Mamba-based classifier for the SDN control plane.

    Takes packet-level flow sequences and outputs attack-class logits.
    Designed to handle "ambiguous" flows escalated from the data-plane CNN.
    """

    def __init__(
        self,
        num_classes: int = 8,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 100,
        input_dim: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.input_dim = input_dim

        # Input embedding: input_dim raw features → d_model
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # Classification head
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, input_dim)
            Packet-level features or tabular features (unsqueezed).

        Returns
        -------
        logits : Tensor (batch, num_classes)
        """
        x = self.input_embedding(x)            # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)                      # (B, d_model)

        x = self.dropout(x)
        logits = self.classifier(x)            # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Focal Loss (for hard-sample emphasis)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced / hard-sample training."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(
                alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
