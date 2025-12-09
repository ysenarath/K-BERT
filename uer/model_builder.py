# -*- encoding:utf-8 -*-
import torch

from uer.encoders.attn_encoder import AttentionEncoder
from uer.encoders.bert_encoder import BERTEncoder
from uer.encoders.birnn_encoder import BiLSTMEncoder
from uer.encoders.cnn_encoder import CNNEncoder, GatedCNNEncoder
from uer.encoders.gpt_encoder import GPTEncoder
from uer.encoders.mixed_encoder import CRNNEncoder, RCNNEncoder
from uer.encoders.rnn_encoder import GRUEncoder, LSTMEncoder
from uer.layers.embeddings import BERTEmbedding
from uer.models.model import Model
from uer.subencoders.avg_subencoder import AvgSubencoder
from uer.subencoders.cnn_subencoder import CNNSubencoder
from uer.subencoders.rnn_subencoder import LSTMSubencoder
from uer.targets.bert_target import BertTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.lm_target import LmTarget
from uer.targets.mlm_target import MlmTarget
from uer.targets.nsp_target import NspTarget
from uer.targets.s2s_target import S2sTarget


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](
            args, len(args.sub_vocab)
        )
    else:
        subencoder = None

    embedding = BERTEmbedding(args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    model = Model(args, embedding, encoder, target, subencoder)

    return model
