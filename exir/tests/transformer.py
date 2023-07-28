# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_embed_dim = (
            2 * embed_dim
        )  # for simplicity we just hardcode ffn_embed_dim to be 2x of embed_dim
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        residual = x
        query = key = value = x
        x, _ = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            add_zero_attn=False,
            dropout_p=0.0,
            use_separate_proj_weight=True,
            in_proj_weight=None,
            in_proj_bias=None,
            # is non None value really needed for bias_k, bias_v?
            bias_k=None,
            bias_v=None,
        )
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


@torch.no_grad()
class Transformer(nn.Module):
    """
    A simplified implementation of mt_model that does not have all those heavy
    dependencies but still be similar enough to the original model.

    Suitable to be put in exir end2end tests. E.g., we can use it to ease the
    testing of memory planning for dynamic shapes on REAL models.

    Some of the simplifications recorded here:
    1. the original model will reset the embedding to a 0 vector for padding token.
       We skip that.
    2. skip various configurations in the original model. E.g., original model
       has a config cfg.no_scale_embedding to control if the token embedding
       should be scaled or not. We just always scale the embedding.
    """

    def __init__(self, inp_vocab_size=10, model_dim=32, num_encoder_layers=2):
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.model_dim = 32
        self.token_embed_table = nn.Embedding(self.inp_vocab_size, self.model_dim)
        self.embed_scale = math.sqrt(self.model_dim)
        self.encoder_layers = [
            EncoderLayer(embed_dim=self.model_dim) for _ in range(num_encoder_layers)
        ]

    def encode(self, src_tokens):
        # embed = self.token_embed_table(src_tokens) * self.embed_scale # fail in runtime because of lacking broadcasting
        embed = self.token_embed_table(src_tokens)
        # TODO: add the support for positional embedding

        # BxTxC -> TxBxC
        x = embed.transpose(0, 1)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def get_random_inputs(self, method):
        if method == "encode":
            seqlen = 10  # TODO: make the sequence length dynamic
            return torch.randint(
                low=0,
                high=self.inp_vocab_size,
                size=(
                    1,
                    seqlen,
                ),
            )
        else:
            raise AssertionError(f"method {method} is not supported yet")
