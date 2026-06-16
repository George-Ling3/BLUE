"""BLUE gate runtime for SimLingo closed-loop evaluation.

The gate is a lightweight binary classifier over the last language-token hidden
state before language generation. It returns 1 when SimLingo should generate
language for the current frame and 0 when SimLingo should directly produce the
driving action.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseGate(nn.Module):
    """Common interface for BLUE gate modules."""

    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return binary decisions for hidden states with shape [B, hidden_size]."""
        raise NotImplementedError


class BlueGate(BaseGate):
    """MLP gate used by BLUE Stage 1 evaluation."""

    def __init__(
        self,
        hidden_size: int = 896,
        mlp_hidden_size: int = 256,
        threshold: float = 0.66,
        dropout: float = 0.0,
        norm_type: str = "none",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__(hidden_size)
        self.threshold = float(threshold)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.dropout = float(dropout)
        self.norm_type = norm_type
        self.classifier = self._build_classifier(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            dropout=self.dropout,
            norm_type=self.norm_type,
        )

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    @staticmethod
    def _build_classifier(
        hidden_size: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        norm_type: str = "none",
    ) -> nn.Sequential:
        layers = [nn.Linear(hidden_size, mlp_hidden_size)]
        if norm_type == "layernorm":
            layers.append(nn.LayerNorm(mlp_hidden_size))
        elif norm_type == "batchnorm":
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
        elif norm_type != "none":
            raise ValueError(f"Unsupported gate norm_type: {norm_type}")
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(mlp_hidden_size, 1))
        return nn.Sequential(*layers)

    def load_checkpoint(self, ckpt_path: str) -> None:
        """Load a BLUE gate checkpoint or a compatible prototype checkpoint."""
        print(f"[BlueGate] Loading gate checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        config = raw.get("config", {}) if isinstance(raw, dict) else {}
        saved_hidden = self._get_saved_hidden_size(config)
        saved_dropout = float(config.get("dropout", self.dropout))
        saved_norm = str(config.get("norm_type", self.norm_type))

        if saved_hidden is not None:
            self.mlp_hidden_size = int(saved_hidden)
            self.dropout = saved_dropout
            self.norm_type = saved_norm
            self.classifier = self._build_classifier(
                hidden_size=self.hidden_size,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                norm_type=self.norm_type,
            )
            print(
                "[BlueGate] Rebuilt classifier from checkpoint config: "
                f"mlp_hidden_size={self.mlp_hidden_size}, dropout={self.dropout}, "
                f"norm_type={self.norm_type}"
            )

        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state_dict = self._normalize_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=True)
        self.eval()
        print("[BlueGate] Gate checkpoint loaded successfully.")

    @staticmethod
    def _get_saved_hidden_size(config: Dict[str, Any]) -> Optional[int]:
        hidden_dims = config.get("hidden_dims")
        if hidden_dims is not None:
            return int(hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else hidden_dims)
        for key in ("hidden_dim", "mlp_hidden_size"):
            if key in config and config[key] is not None:
                return int(config[key])
        return None

    @staticmethod
    def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "gate."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            normalized[new_key] = value
        return normalized

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        decisions, _ = self.forward_with_prob(hidden_state)
        return decisions

    def forward_with_prob(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(hidden_state)
        probs = torch.sigmoid(logits).squeeze(-1)
        decisions = (probs >= self.threshold).long()
        return decisions, probs


def create_gate(
    mode: str = "trained_gate",
    hidden_size: int = 896,
    ckpt_path: Optional[str] = None,
    **kwargs: Any,
) -> BaseGate:
    """Create the public BLUE gate.

    Stage 1 intentionally exposes only ``trained_gate``. Prototype aliases are
    accepted only for checkpoint/config backward compatibility and should not be
    shown in public scripts or documentation.
    """
    if mode not in {"trained_gate", "trained"}:
        raise ValueError("BLUE Stage 1 only supports mode='trained_gate'.")
    if ckpt_path is None:
        raise ValueError("A BLUE gate checkpoint is required for trained_gate mode.")
    return BlueGate(
        hidden_size=hidden_size,
        mlp_hidden_size=int(kwargs.get("mlp_hidden_size", 256)),
        threshold=float(kwargs.get("threshold", 0.66)),
        dropout=float(kwargs.get("dropout", 0.0)),
        norm_type=str(kwargs.get("norm_type", "none")),
        ckpt_path=ckpt_path,
    )
