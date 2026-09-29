#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class SamplingHead(nn.Module):
    """
    Wraps a model that returns last-token logits ``(B, vocab)`` and samples a
    token id ``(B)`` on-device.

        forward(*model_args, temperature, top_k, top_p, seed) -> token_id

    The sampling params are trailing positional args so the head is directly
    exportable (``torch.export`` drives positional inputs) without a per-model
    wrapper.

      temperature: scalar float tensor, e.g. torch.tensor(0.8). Must be >= 0;
                   temperature=0 is greedy (returns argmax, no division).
      top_k:       scalar int tensor; keeps only the k most likely tokens. Use
                   the max int (clipped to the vocab size) to keep every token.
      top_p:       scalar float tensor in (0, 1] for nucleus sampling. top_p=1.0
                   keeps every token, i.e. no filtering.
      seed:        scalar int tensor (seeded) or None (unseeded export)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args):
        *model_args, temperature, top_k, top_p, seed = args
        logits = self.model(*model_args)  # [B, vocab]
        if not isinstance(top_k, torch.Tensor):
            top_k = torch.tensor(int(top_k), dtype=torch.int64)
        if not isinstance(top_p, torch.Tensor):
            top_p = torch.tensor(float(top_p))
        return torch.ops.mlx.sample(logits, temperature, top_k, top_p, seed)
