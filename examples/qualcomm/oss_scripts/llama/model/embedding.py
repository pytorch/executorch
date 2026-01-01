# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class TextEmbedding(nn.Module):
    def __init__(
        self,
        input_embedding_module,
        max_batch_size: int,
        ar_len: int,
        vocab_size: int,
        dim: int,
        use_i64_token: bool,
    ):
        super().__init__()
        self.input_embedding_module = input_embedding_module
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_i64_token = use_i64_token

    def get_example_input(self):
        return (
            torch.randint(
                self.vocab_size,
                (self.max_batch_size, self.ar_len),
                dtype=torch.int64 if self.use_i64_token else torch.int32,
            ),
        )

    def forward(self, input_ids):
        return self.input_embedding_module(input_ids)
