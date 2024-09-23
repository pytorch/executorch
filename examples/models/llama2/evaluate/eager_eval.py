# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Union

import lm_eval
import torch
from executorch.examples.models.llama2.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.extension.llm.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)

from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict

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
        super().__init__(device=device)
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self._use_kv_cache = use_kv_cache

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

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

    def tok_encode(self, string: str, **kwargs):
        tokens = self._tokenizer.encode(string, bos=True, eos=False)
        encoded = torch.tensor(tokens, dtype=torch.int, device=self.device)
        # encoded is a pytorch tensor, but some internal logic in the
        # eval harness expects it to be a list instead
        # TODO: verify this for multi-batch as well
        encoded = encoded.tolist()
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

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


@torch.no_grad()
def evaluate_model(
    eval_wrapper: LM,
    tasks: Optional[list] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        eval_wrapper (LM): A LM wrapper class compatible with lm-evaluation-harness evaluation
        tasks: Optional[list]: The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """

    if tasks is None:
        tasks = ["wikitext"]

    if "hendrycks_test" in tasks:
        tasks.remove("hendrycks_test")
        tasks += list(lm_eval.tasks.hendrycks_test.create_all_tasks().keys())
    task_dict = get_task_dict(tasks)

    eval_results = evaluate(
        eval_wrapper,
        task_dict,
        limit=limit,
    )
    return eval_results
