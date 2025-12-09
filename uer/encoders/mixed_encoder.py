# -*- encoding:utf-8 -*-
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RCNNEncoderConfig:
    emb_size: int = 512
    hidden_size: int = 512
    kernel_size: int = 3
    layers_num: int = 2
    dropout: float = 0.1


class RCNNEncoder(nn.Module):
    def __init__(self, config: RCNNEncoderConfig):
        super(RCNNEncoder, self).__init__()

        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.layers_num = config.layers_num

        self.rnn = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=config.hidden_size,
            num_layers=config.layers_num,
            dropout=config.dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(config.dropout)

        self.conv_1 = nn.Conv2d(
            1, config.hidden_size, (config.kernel_size, config.emb_size)
        )
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(
                    config.hidden_size, config.hidden_size, (config.kernel_size, 1)
                )
                for _ in range(config.layers_num - 1)
            ]
        )

    def forward(self, emb: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emb.size()

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size]).to(
            emb.device
        )
        hidden = torch.cat([padding, output], dim=1).unsqueeze(
            1
        )  # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(hidden)
        padding = torch.zeros(
            [batch_size, self.hidden_size, self.kernel_size - 1, 1]
        ).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1 :, :]
        output = (
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        return output

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
        )


@dataclass
class CRNNEncoderConfig:
    emb_size: int = 512
    hidden_size: int = 512
    kernel_size: int = 3
    layers_num: int = 2
    dropout: float = 0.1


class CRNNEncoder(nn.Module):
    def __init__(self, config: CRNNEncoderConfig):
        super(CRNNEncoder, self).__init__()

        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.layers_num = config.layers_num

        self.conv_1 = nn.Conv2d(
            1, config.hidden_size, (config.kernel_size, config.emb_size)
        )
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(
                    config.hidden_size, config.hidden_size, (config.kernel_size, 1)
                )
                for _ in range(config.layers_num - 1)
            ]
        )
        self.rnn = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=config.hidden_size,
            num_layers=config.layers_num,
            dropout=config.dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, emb: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emb.size()
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size]).to(
            emb.device
        )
        emb = torch.cat([padding, emb], dim=1).unsqueeze(
            1
        )  # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(emb)
        padding = torch.zeros(
            [batch_size, self.hidden_size, self.kernel_size - 1, 1]
        ).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1 :, :]
        output = (
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(output, hidden)
        output = self.drop(output)

        return output

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
            torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
        )
