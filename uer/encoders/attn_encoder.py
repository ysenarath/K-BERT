# -*- encoding:utf-8 -*-
from dataclasses import dataclass

import torch
import torch.nn as nn

from uer.layers.multi_headed_attn import MultiHeadedAttention


@dataclass
class AttentionEncoderConfig:
    layers_num: int = 12
    heads_num: int = 12
    hidden_size: int = 768
    dropout: float = 0.1


class AttentionEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, config: AttentionEncoderConfig):
        super(AttentionEncoder, self).__init__()
        self.self_attn = MultiHeadedAttention(
            config.hidden_size, config.heads_num, config.dropout
        )
        self.self_attn = nn.ModuleList(
            [
                MultiHeadedAttention(
                    config.hidden_size, config.heads_num, config.dropout
                )
                for _ in range(config.layers_num)
            ]
        )

    @property
    def layers_num(self):
        return len(self.self_attn)

    def forward(self, emb: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.self_attn[i](hidden, hidden, hidden, mask)

        return hidden
