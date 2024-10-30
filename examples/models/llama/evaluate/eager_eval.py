# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Union

import torch
from executorch.examples.models.llama.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.extension.llm.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)

from lm_eval.models.huggingface import HFLM as eval_wrapper

from torch import nn


class EagerEvalWrapper(eval_wrapper):
    """
    A wrapper class based on GPTFast, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        max_seq_length: Optional[int] = None,
        use_kv_cache: bool = False,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device=device, pretrained="gpt2")
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self._use_kv_cache = use_kv_cache

    @property
    def eot_token_id(self):
        """
        The stories model does not have an EOT token, so we use the EOS token instead.
        """
        if hasattr(self._tokenizer, "eot_id"):
            return self._tokenizer.eot_id
        return self._tokenizer.eos_id

    @property
    def prefix_token_id(self):
        return self.eot_token_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):  # pyre-ignore
        return self._tokenizer.encode(string, bos=False, eos=False)

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps):
        if self._use_kv_cache:
            pos_tensor = torch.tensor([0], dtype=torch.int64, device=self.device)
            # Batch process the whole sequence.
            logits = self._model(inps[:, : self._max_seq_length], pos_tensor)
            return logits
        else:
            return self._model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")
