from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torchvision


def _as_hidden_dims(hidden_dims: Iterable[int] | str | None) -> list[int]:
    if hidden_dims is None:
        return [256]
    if isinstance(hidden_dims, str):
        if hidden_dims.strip() == "":
            return []
        return [int(v.strip()) for v in hidden_dims.split(",") if v.strip()]
    return [int(v) for v in hidden_dims]


class EchoPrimeBinaryClassifier(nn.Module):
    """EchoPrime video encoder plus a small binary classification head."""

    def __init__(
        self,
        weights_path: str | Path,
        hidden_dims: Iterable[int] | str | None = (256,),
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.weights_path = str(weights_path)
        self.embedding_dim = embedding_dim
        self.hidden_dims = _as_hidden_dims(hidden_dims)
        self.dropout = float(dropout)
        self.freeze_encoder = bool(freeze_encoder)

        self.encoder = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = nn.Linear(
            self.encoder.head[-1].in_features, embedding_dim
        )
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.encoder.load_state_dict(checkpoint)

        self.classifier = self._build_classifier()
        self.set_encoder_trainable(not self.freeze_encoder)

    def _build_classifier(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = self.embedding_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.freeze_encoder = not trainable
        for param in self.encoder.parameters():
            param.requires_grad = trainable
        if self.freeze_encoder:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(video)
        logits = self.classifier(features).squeeze(-1)
        return logits

    def config(self) -> dict:
        return {
            "weights_path": self.weights_path,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "freeze_encoder": self.freeze_encoder,
            "embedding_dim": self.embedding_dim,
        }
