# -*- encoding:utf-8 -*-
from dataclasses import dataclass

import torch.nn as nn

from uer.layers.layer_norm import LayerNorm
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.position_ffn import PositionwiseFeedForward


@dataclass
class TransformerLayerConfig:
    hidden_size: int = 768
    heads_num: int = 12
    feedforward_size: int = 3072
    dropout: float = 0.1


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, config: TransformerLayerConfig):
        super(TransformerLayer, self).__init__()
        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            config.hidden_size, config.heads_num, config.dropout
        )
        self.dropout_1 = nn.Dropout(config.dropout)
        self.layer_norm_1 = LayerNorm(config.hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            config.hidden_size, config.feedforward_size
        )
        self.dropout_2 = nn.Dropout(config.dropout)
        self.layer_norm_2 = LayerNorm(config.hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output
