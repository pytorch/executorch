# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from typing import Optional

import lm_eval

import torch
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict
from sentencepiece import SentencePieceProcessor
from torch import nn

from .builder import LlamaEdgeManager
from .export_llama_lib import (
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
)


class GPTFastEvalWrapper(eval_wrapper):
    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

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
        tokens = [self._tokenizer.bos_id()] + self._tokenizer.encode(string)
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
        return self._model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


@torch.no_grad()
def eval(
    model: nn.Module,
    tokenizer,
    tasks: Optional[list] = None,
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        model (nn.Module): The pre-trained language model to evaluate.
        tokenizer: The tokenizer to use for encoding/decoding text.
        task (str): The name of the evaluation task to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """

    if tasks is None:
        tasks = ["wikitext"]

    model_eval_wrapper = GPTFastEvalWrapper(
        model,
        tokenizer,
        max_seq_length,
    )

    if "hendrycks_test" in tasks:
        tasks.remove("hendrycks_test")
        tasks += list(lm_eval.tasks.hendrycks_test.create_all_tasks().keys())
    task_dict = get_task_dict(tasks)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
    )
    return eval_results


def build_args_parser() -> argparse.ArgumentParser:
    # Start with arg parser from export_llama_lib
    parser = _build_args_parser()

    # Add additional args specific to eval
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="number of samples to evalulate"
    )

    return parser


def eval_llama(
    model_name: str,
    args: argparse.ArgumentParser,
) -> None:
    # Get a pre-lowering/to_edge LlamaEdgeManager instance
    manager: LlamaEdgeManager = _prepare_for_llama_export(model_name, args)
    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer_path))

    # Evaluate the model
    eval_results = eval(
        manager.model.to(device="cpu"),
        tokenizer,
        args.tasks,
        args.limit,
        args.max_seq_length,
    )

    print("Results: ", eval_results)
