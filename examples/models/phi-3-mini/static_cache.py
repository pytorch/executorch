# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from transformers import PretrainedConfig, StaticCache


class ETStaticCache(StaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum().item()

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)
