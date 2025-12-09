# -*- encoding:utf-8 -*-
from dataclasses import dataclass
import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer, TransformerLayerConfig


@dataclass
class GPTEncoderConfig(TransformerLayerConfig):
    layers_num: int = 12


class GPTEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, config: GPTEncoderConfig):
        super(GPTEncoder, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerLayer(config) for _ in range(self.layers_num)]
        )

    @property
    def layers_num(self):
        return len(self.transformer)

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        mask = torch.ones(seq_length, seq_length, device=emb.device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = mask.repeat(batch_size, 1, 1, 1)

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden
