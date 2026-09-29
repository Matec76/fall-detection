"""Classifiers over pose sequences.

``rnn`` / ``lstm`` / ``gru``: one recurrent hidden layer followed by a fully connected
layer and softmax, as in Lin et al. 2021 (Fig. 13, Table 4).
``stacked_lstm`` / ``transformer``: extra baselines from Juraev et al. 2022.
``tcn``: a small dilated temporal convolution network (not in either paper).
Recurrent models also take ``bidirectional: true`` and ``layers: n``.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    """One recurrent layer (tanh RNN, LSTM or GRU) whose last hidden state feeds a
    fully connected output layer. Lin et al. tried 64-1024 hidden nodes; 512 was best."""

    CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, in_dim: int, num_classes: int, seq_len: int | None = None,
                 cell: str = "lstm", hidden: int = 512, dropout: float = 0.0,
                 bidirectional: bool = False, layers: int = 1):
        super().__init__()
        self.rnn = self.CELLS[cell](in_dim, hidden, num_layers=layers, batch_first=True,
                                    bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        out, _ = self.rnn(x)
        if self.bidirectional:  # last step of the forward pass, first step of the backward pass
            half = out.shape[-1] // 2
            last = torch.cat([out[:, -1, :half], out[:, 0, half:]], dim=-1)
        else:
            last = out[:, -1]
        return self.fc(self.dropout(last))


class TemporalCNN(nn.Module):
    """Stacked dilated 1-D convolutions over time, then average + max pooling."""

    def __init__(self, in_dim: int, num_classes: int, seq_len: int | None = None,
                 channels: int = 64, kernel: int = 5, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, prev = [], in_dim
        for i in range(layers):
            blocks += [nn.Conv1d(prev, channels, kernel, padding=(kernel // 2) * 2 ** i, dilation=2 ** i),
                       nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout)]
            prev = channels
        self.net = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        h = self.net(x.transpose(1, 2))
        return self.fc(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


class PoseTransformer(nn.Module):
    """Action-Transformer-style encoder: linear projection of per-frame keypoints,
    a class token plus learned positional embeddings, then an MLP head (Fig. 2).
    Defaults (4 layers, 1 head, d_model 64) give ~0.22M parameters as in Table 6.
    """

    def __init__(self, in_dim: int, num_classes: int, seq_len: int, d_model: int = 64,
                 nhead: int = 1, num_layers: int = 4, dim_ff: int = 256, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim_ff, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        x = self.proj(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1]
        x = self.encoder(self.dropout(x))
        return self.head(self.norm(x[:, 0]))


class PoseLSTM(nn.Module):
    """Stacked LSTMs (32-64-32 units) followed by dense layers (128-64-32-16), Fig. 3.
    Defaults give ~0.061M parameters as in Table 6.
    """

    def __init__(self, in_dim: int, num_classes: int, seq_len: int | None = None,
                 hidden: tuple[int, ...] = (32, 64, 32), dense: tuple[int, ...] = (128, 64, 32, 16),
                 dropout: float = 0.3):
        super().__init__()
        self.lstms = nn.ModuleList()
        prev = in_dim
        for size in hidden:
            self.lstms.append(nn.LSTM(prev, size, batch_first=True))
            prev = size
        self.dropout = nn.Dropout(dropout)
        layers: list[nn.Module] = []
        for size in dense:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(dropout)]
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, in_dim)
        for lstm in self.lstms:
            x, _ = lstm(x)
            x = self.dropout(x)
        return self.head(x[:, -1])


MODELS = {"transformer": PoseTransformer, "stacked_lstm": PoseLSTM, "tcn": TemporalCNN}
RECURRENT = ("rnn", "lstm", "gru")


def build_model(name: str, in_dim: int, num_classes: int, seq_len: int, **kwargs) -> nn.Module:
    if name in RECURRENT:
        return RecurrentClassifier(in_dim, num_classes, seq_len, cell=name, **kwargs)
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted([*RECURRENT, *MODELS])}")
    return MODELS[name](in_dim=in_dim, num_classes=num_classes, seq_len=seq_len, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint written by ``falldet.train``."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], ckpt["in_dim"], len(ckpt["classes"]),
                        ckpt["preprocess"]["seq_len"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
