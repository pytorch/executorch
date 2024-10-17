# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn
from transformers import Phi3ForCausalLM

from .static_cache import ETStaticCache


class Phi3Mini(torch.nn.Module):

    def __init__(self, model: Phi3ForCausalLM, max_batch_size: int, max_seq_len: int):
        super().__init__()
        self.model = model
        self.cache = ETStaticCache(
            # pyre-fixme[16]: `Phi3ForCausalLM` has no attribute `config`.
            config=model.config,
            max_batch_size=max_batch_size,
            max_cache_len=max_seq_len,
            # pyre-fixme[16]: `Phi3ForCausalLM` has no attribute `device`.
            device=self.model.device,
            # pyre-fixme[16]: `Phi3ForCausalLM` has no attribute `dtype`.
            dtype=self.model.dtype,
        )

    def forward(
        self,
        # pyre-fixme[9]: input_ids has type `LongTensor`; used as `None`.
        input_ids: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        # pyre-fixme[16]: `Phi3ForCausalLM` has no attribute `forward`.
        return self.model.forward(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            past_key_values=self.cache,
        ).logits[:, -1, :]
