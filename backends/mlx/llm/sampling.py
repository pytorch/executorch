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
    Wraps a model that returns logits and samples a token id on-device.

        forward(*model_args, temperature, top_k=None, top_p=1.0, seed=None,
                **model_kwargs) -> token_id

      temperature: scalar float tensor, e.g. torch.tensor(0.8). Must be >= 0;
                   temperature=0 is greedy (returns argmax, no division).
      top_k:       scalar int tensor or int; keeps only the k most likely tokens.
                   None uses the max int default, which is clipped to the vocab
                   size and keeps every token.
      top_p:       scalar float tensor in (0, 1] for nucleus sampling. top_p=1.0
                   (the default) keeps every token, i.e. no filtering. Pass it
                   as a runtime input to tune per request.
      seed:        scalar int tensor (seeded) or None (unseeded export)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, temperature, top_k=None, top_p=1.0, seed=None, **kwargs):
        logits = self.model(*args, **kwargs)  # [B, S, vocab]
        last = logits[:, -1, :]  # [B, vocab]
        if not isinstance(top_p, torch.Tensor):
            top_p = torch.tensor(float(top_p))
        if top_k is None:
            top_k = torch.tensor(torch.iinfo(torch.int64).max, dtype=torch.int64)
        elif not isinstance(top_k, torch.Tensor):
            top_k = torch.tensor(int(top_k), dtype=torch.int64)
        return torch.ops.mlx.sample(last, temperature, top_k, top_p, seed)
