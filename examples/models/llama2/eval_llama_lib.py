# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

from typing import Optional, Union

import lm_eval
import torch

from executorch.examples.models.llama2.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.examples.models.llama2.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)

from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict

from torch import nn

from .builder import LlamaEdgeManager
from .export_llama_lib import (
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
)


class GPTFastEvalWrapper(eval_wrapper):
    """
    A wrapper class based on GPTFast, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

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
        return self._model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


class ETEagerEvalWrapper(GPTFastEvalWrapper):
    """
    A wrapper class for ExecuTorch Eager integration with the
    lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        max_seq_length: Optional[int] = None,
    ):
        super().__init__(None, tokenizer, max_seq_length)
        self._model = model  # Expects model to be path to a .pte file

        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        self._et_model = _load_for_executorch(self._model)

    def _model_call(self, inps):
        # Given inps (tokens), return the logits from a single forward call
        # inps: Tensor of shape (1, max_seq_len - 1)
        # logits: Tensor of shape (1, max_seq_len - 1, 32000)
        result = self._et_model.forward((inps,))
        return result[0]


class ETRunnerEvalWrapper(GPTFastEvalWrapper):
    """
    A wrapper class for ExecuTorch Runtime integration with the
    lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        tokenizer_bin: str,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__(None, tokenizer, max_seq_length)
        self._model = model
        self._tokenizer_bin = tokenizer_bin

    def _model_call(self, inps):
        # Given inps (tokens), return the logits from a single
        # forward call

        # Example:
        # inps: Tensor of shape (1, N)
        # logits: Tensor of shape (1, N, 32000)
        pass


@torch.no_grad()
def eval(
    eval_wrapper: LM,
    tasks: Optional[list] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        eval_wrapper (LM): A LM wrapper class compatible with lm-evaluation-harness evaluation
        task (str): The name of the evaluation task to perform.
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


def gen_eval_wrapper(
    model_name: str,
    args: argparse.ArgumentParser,
) -> LM:
    """
    Generates a wrapper interface around the provided model and tokenizer for
    the lm-evaluation-harness library.

    Returns:
        eval_wrapper (LM): A wrapper interface for the lm-evaluation-harness library.
    """
    try:
        tokenizer = SentencePieceTokenizer(model_path=str(args.tokenizer_path))
    except Exception:
        print("Using Tiktokenizer")
        tokenizer = Tiktoken(model_path=str(args.tokenizer_path))

    # ExecuTorch Binary Evaluation
    if (model := args.pte) is not None:
        if (tokenizer_bin := args.tokenizer_bin) is not None:
            # ETRunnerEvalWrapper: Create a wrapper around an ExecuTorch model, evaluated at runtime
            return ETRunnerEvalWrapper(
                model=model,
                tokenizer=tokenizer,
                tokenizer_bin=tokenizer_bin,
                max_seq_length=args.max_seq_length,
            )

        # ETRunnerEvalWrapper: Create a wrapper around an ExecuTorch model, evaluated eagerly
        return ETEagerEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            # Exported model takes at most (max_seq_length - 1) tokens.
            # Note that the eager model takes at most max_seq_length tokens.
            max_seq_length=args.max_seq_length - 1,
        )

    # GPTFastEvalWrapper: Create a wrapper around a pre-exported model
    manager: LlamaEdgeManager = _prepare_for_llama_export(model_name, args)
    model = (
        manager.model.eval().to(device="cuda")
        if torch.cuda.is_available()
        else manager.model.to(device="cpu")
    )
    return GPTFastEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )


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

    # Add additional args specific to eval via an ET Runner
    # Note: For initial integration, the tokenizer.model is also required
    parser.add_argument(
        "--pte",
        type=str,
        default=None,
        help="[For ExecuTorch] Path to the ExecuTorch model being evaluated. If provided, don't go through the export flow",
    )
    parser.add_argument(
        "--tokenizer_bin",
        type=str,
        default=None,
        help="[For ExecuTorch] Path to the Tokenizer binary for evaluating ExecuTorch models via runtime",
    )

    return parser


def eval_llama(
    model_name: str,
    args: argparse.ArgumentParser,
) -> None:
    # Generate the eval wrapper
    eval_wrapper = gen_eval_wrapper(model_name, args)

    # Evaluate the model
    eval_results = eval(
        eval_wrapper,
        args.tasks,
        args.limit,
    )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")
