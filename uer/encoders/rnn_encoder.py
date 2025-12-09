# -*- encoding:utf-8 -*-
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LSTMEncoderConfig:
    emb_size: int
    hidden_size: int
    layers_num: int
    dropout: float
    bidirectional: bool


class LSTMEncoder(nn.Module):
    def __init__(self, config: LSTMEncoderConfig):
        super(LSTMEncoder, self).__init__()

        self.bidirectional = config.bidirectional
        if self.bidirectional:
            assert config.hidden_size % 2 == 0
            self.hidden_size = config.hidden_size // 2
        else:
            self.hidden_size = config.hidden_size

        self.layers_num = config.layers_num

        self.rnn = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=self.hidden_size,
            num_layers=config.layers_num,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.drop = nn.Dropout(config.dropout)

    def forward(self, emb: torch.Tensor, seg: torch.Tensor):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.bidirectional:
            return (
                torch.zeros(
                    self.layers_num * 2, batch_size, self.hidden_size, device=device
                ),
                torch.zeros(
                    self.layers_num * 2, batch_size, self.hidden_size, device=device
                ),
            )
        else:
            return (
                torch.zeros(
                    self.layers_num, batch_size, self.hidden_size, device=device
                ),
                torch.zeros(
                    self.layers_num, batch_size, self.hidden_size, device=device
                ),
            )


@dataclass
class GRUEncoderConfig:
    emb_size: int
    hidden_size: int
    layers_num: int
    dropout: float
    bidirectional: bool


class GRUEncoder(nn.Module):
    def __init__(self, args: GRUEncoderConfig):
        super(GRUEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0
            self.hidden_size = args.hidden_size // 2
        else:
            self.hidden_size = args.hidden_size

        self.layers_num = args.layers_num

        self.rnn = nn.GRU(
            input_size=args.emb_size,
            hidden_size=self.hidden_size,
            num_layers=args.layers_num,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb: torch.Tensor, seg: torch.Tensor):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.bidirectional:
            return torch.zeros(
                self.layers_num * 2, batch_size, self.hidden_size, device=device
            )
        else:
            return torch.zeros(
                self.layers_num, batch_size, self.hidden_size, device=device
            )
