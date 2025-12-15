# -*- encoding:utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from uer.layers.layer_norm import LayerNorm


@dataclass
class BERTEmbeddingConfig:
    emb_size: int = 768
    dropout: float = 0.1


class BERTEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """

    def __init__(self, config: BERTEmbeddingConfig, vocab_size: int):
        super(BERTEmbedding, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, config.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, config.emb_size)
        self.segment_embedding = nn.Embedding(3, config.emb_size)
        self.layer_norm = LayerNorm(config.emb_size)

    def forward(
        self, src: torch.Tensor, seg: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(
                torch.arange(
                    0, word_emb.size(1), device=word_emb.device, dtype=torch.long
                )
                .unsqueeze(0)
                .repeat(word_emb.size(0), 1)
            )
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb
