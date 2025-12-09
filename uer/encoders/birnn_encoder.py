# -*- encoding:utf-8 -*-
from dataclasses import dataclass

import torch
import torch.nn as nn


def flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


@dataclass
class BiLSTMEncoderConfig:
    emb_size: int = 768
    hidden_size: int = 768
    layers_num: int = 2
    dropout: float = 0.1


class BiLSTMEncoder(nn.Module):
    def __init__(self, config: BiLSTMEncoderConfig):
        super(BiLSTMEncoder, self).__init__()

        assert config.hidden_size % 2 == 0
        self.hidden_size = config.hidden_size // 2

        self.layers_num = config.layers_num

        self.rnn_forward = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=self.hidden_size,
            num_layers=config.layers_num,
            dropout=config.dropout,
            batch_first=True,
        )

        self.rnn_backward = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=self.hidden_size,
            num_layers=config.layers_num,
            dropout=config.dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(config.dropout)

    def forward(self, emb, seg):
        # Forward.
        emb_forward = emb
        hidden_forward = self.init_hidden(emb_forward.size(0), emb_forward.device)
        output_forward, hidden_forward = self.rnn_forward(emb_forward, hidden_forward)
        output_forward = self.drop(output_forward)

        # Backward.
        emb_backward = flip(emb, 1)
        hidden_backward = self.init_hidden(emb_backward.size(0), emb_backward.device)
        output_backward, hidden_backward = self.rnn_backward(
            emb_backward, hidden_backward
        )
        output_backward = self.drop(output_backward)

        return torch.cat([output_forward, output_backward], 2)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
        )
