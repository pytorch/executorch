# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

from typing import Optional, Union

import torch
from executorch.examples.models.llama2.evaluate import EagerEvalWrapper, evaluate_model
from executorch.examples.models.llama2.export_llama_lib import (
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama2.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.examples.models.llama2.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)

from lm_eval.api.model import LM

from .builder import LlamaEdgeManager
from .export_llama_lib import (
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
)


class ETPybindEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class for ExecuTorch py-binded integration with the
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
        self._use_kv_cache = self._et_model.run_method("use_kv_cache")[0]

    def _model_call(self, inps):
        # Given inps (tokens), return the logits from a single forward call
        # inps: Tensor of shape (1, max_seq_len - 1)
        # logits: Tensor of shape (1, max_seq_len - 1, vocab_size)
        if self._use_kv_cache:
            result_logits = []
            for pos in range(self._max_seq_length):
                pos_tensor = torch.tensor([pos], dtype=torch.int64)
                logits = self._et_model.forward((inps[:, pos : pos + 1], pos_tensor))
                result_logits.append(logits[0])
            return torch.cat(result_logits, dim=1)
        else:
            result = self._et_model.forward((inps,))
            return result[0]


class ETRunnerEvalWrapper(EagerEvalWrapper):
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
        # logits: Tensor of shape (1, N, vocab_size)
        pass


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

        # ETPybindEvalWrapper: Create a wrapper around an ExecuTorch model, evaluated with pybindings
        return ETPybindEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            # Exported model takes at most (max_seq_length - 1) tokens.
            # Note that the eager model takes at most max_seq_length tokens.
            max_seq_length=args.max_seq_length - 1,
        )

    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)
    # GPTFastEvalWrapper: Create a wrapper around a pre-exported model
    manager: LlamaEdgeManager = _prepare_for_llama_export(model_name, args)

    if len(quantizers) != 0:
        manager = manager.capture_pre_autograd_graph().pt2e_quantize(quantizers)
        model = (
            manager.pre_autograd_graph_module.to(device="cuda")
            if torch.cuda.is_available()
            else manager.pre_autograd_graph_module.to(device="cpu")
        )
    else:
        # TODO: use manager.pre_autograd_graph_module for the eval to remove the if-else branch
        # for quantizers. Currently capture_pre_autograd_graph only works with --kv_cache, but
        # fails without the kv_cache mode
        model = (
            manager.model.eval().to(device="cuda")
            if torch.cuda.is_available()
            else manager.model.eval().to(device="cpu")
        )

    return EagerEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        use_kv_cache=args.use_kv_cache,
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
    eval_results = evaluate_model(
        eval_wrapper,
        args.tasks,
        args.limit,
    )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")
