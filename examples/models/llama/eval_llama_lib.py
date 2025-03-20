# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Optional, Union

import torch
from executorch.examples.models.llama.export_llama_lib import (
    _convert_args_to_config,
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.extension.llm.export.builder import LLMEdgeManager
from executorch.extension.llm.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)
from executorch.extension.llm.tokenizer.utils import get_tokenizer
from lm_eval.evaluator import simple_evaluate
from omegaconf import DictConfig, OmegaConf

from .evaluate.eager_eval import EagerEvalWrapper


class GraphModuleEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class for ExecuTorch py-binded integration with the
    lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: torch.fx.GraphModule,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        max_seq_length: Optional[int] = None,
        use_kv_cache: bool = False,
        generate_full_logits: bool = False,
        enable_dynamic_shape: bool = True,
    ):
        super().__init__(
            model=model, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
        self._model = model.to(self.device)
        self._use_kv_cache = use_kv_cache
        self._generate_full_logits = generate_full_logits
        self._enable_dynamic_shape = enable_dynamic_shape

    def _model_call(self, inps):
        if self._use_kv_cache:
            if not self._enable_dynamic_shape:
                # graph module exported without dynamic shape won't work with a different shape.
                # And we have to do single token prefill here.
                result_logits = []
                for pos in range(inps.shape[-1]):
                    pos_tensor = torch.tensor([pos], dtype=torch.int64)
                    logits = self._model(
                        inps[:, pos : pos + 1], {"input_pos": pos_tensor}
                    )
                    result_logits.append(logits)
                if self._generate_full_logits:
                    return torch.cat(result_logits, dim=1)
                else:
                    return torch.stack(result_logits, dim=1)
            else:
                pos_tensor = torch.tensor([0], dtype=torch.int64, device=self.device)
                # Batch process the whole sequence.
                logits = self._model(
                    inps[:, : self._max_seq_length], {"input_pos": pos_tensor}
                )
                return logits

        else:
            return self._model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


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
        super().__init__(None, tokenizer, max_seq_length)  # pyre-ignore
        self._model = model  # Expects model to be path to a .pte file

        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        # Load custom ops and quantized ops.
        from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

        # Note: import this after portable_lib
        from executorch.extension.llm.custom_ops import (  # noqa
            custom_ops,  # usort: skip
        )
        from executorch.kernels import quantized  # noqa

        self._et_model = _load_for_executorch(self._model)
        self._use_kv_cache = self._et_model.run_method("use_kv_cache")[0]  # pyre-ignore

    def _model_call(self, inps):
        # Given inps (tokens), return the logits from a single forward call
        # inps: Tensor of shape (1, max_seq_len - 1)
        # logits: Tensor of shape (1, max_seq_len - 1, vocab_size)
        result = []
        if self._use_kv_cache:
            pos_tensor = torch.tensor([0], dtype=torch.int64, device=self.device)
            result = self._et_model.forward(
                (inps[:, : self._max_seq_length], pos_tensor)
            )
        else:
            result = self._et_model.forward((inps,))
        if result[0].dim() != 3:
            raise ValueError(
                f"Dim of logits must be 3 for evaluation. Got {result[0].dim()} here. Add --generate_full_logits in export_llama to generate a pte file with full logits."
            )
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
        super().__init__(None, tokenizer, max_seq_length)  # pyre-ignore
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
    config: DictConfig,
):
    """
    Generates a wrapper interface around the provided model and tokenizer for
    the lm-evaluation-harness library.

    Returns:
        eval_wrapper (LM): A wrapper interface for the lm-evaluation-harness library.
    """
    tokenizer = get_tokenizer(config.export.tokenizer_path)

    # ExecuTorch Binary Evaluation
    if (model := config.eval.pte) is not None:
        if (tokenizer_bin := config.eval.tokenizer_bin) is not None:
            # ETRunnerEvalWrapper: Create a wrapper around an ExecuTorch model, evaluated at runtime
            return ETRunnerEvalWrapper(
                model=model,
                tokenizer=tokenizer,
                tokenizer_bin=tokenizer_bin,
                max_seq_length=config.sequence.max_seq_length,
            )

        # ETPybindEvalWrapper: Create a wrapper around an ExecuTorch model, evaluated with pybindings
        return ETPybindEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            # Exported model takes at most (max_seq_length - 1) tokens.
            # Note that the eager model takes at most max_seq_length tokens.
            max_seq_length=config.sequence.max_seq_length - 1,
        )

    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(config)
    # GPTFastEvalWrapper: Create a wrapper around a pre-exported model
    manager: LLMEdgeManager = _prepare_for_llama_export(config)

    if len(quantizers) != 0:
        manager = manager.export().pt2e_quantize(quantizers)
        model = (
            manager.pre_autograd_graph_module.to(device="cuda")  # pyre-ignore
            if torch.cuda.is_available()
            else manager.pre_autograd_graph_module.to(device="cpu")
        )
        return GraphModuleEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=config.sequence.max_seq_length,
            use_kv_cache=config.kv_cache.use_kv_cache,  # pyre-ignore
            enable_dynamic_shape=config.misc.enable_dynamic_shape,  # pyre-ignore
        )
    else:
        # TODO: use manager.pre_autograd_graph_module for the eval to remove the if-else branch
        # for quantizers. Currently export_for_training only works with --kv_cache, but
        # fails without the kv_cache mode
        model = (
            manager.model.eval().to(device="cuda")
            if torch.cuda.is_available()
            else manager.model.eval().to(device="cpu")
        )

        # Save the checkpoint after the eager model preparation is done.
        # The reason for this option is that the checkpoint can be used
        # to do evaluations in other evaluation platforms, or with data
        # that is not available in this eval_llama. We save the checkpoint
        # here for consistency with eval_llama. The accuracy results we
        # get from eval_llama can be used as a reference to other evaluations.
        if config.eval.output_eager_checkpoint_file is not None:  # pyre-ignore
            torch.save(model, config.eval.output_eager_checkpoint_file)

        return EagerEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=config.sequence.max_seq_length,
            use_kv_cache=config.kv_cache.use_kv_cache,
        )


def eval_llama(
    model_name: str,
    config: DictConfig,
) -> None:
    # Generate the eval wrapper
    eval_wrapper = gen_eval_wrapper(model_name, config)

    # Needed for loading mmlu dataset.
    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1998/files
    if config.eval.tasks and "mmlu" in config.eval.tasks:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    # Evaluate the model
    tasks = (
        None if config.eval.tasks is None else OmegaConf.to_container(config.eval.tasks)
    )
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=eval_wrapper,
            tasks=tasks,
            num_fewshot=config.eval.num_fewshot,
            limit=config.eval.limit,
        )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


def eval_llama_with_attention_sink(
    model_name: str,
    config: DictConfig,
) -> None:
    # Generate the eval wrapper
    eval_wrapper = gen_eval_wrapper(model_name, config)

    # Needed for loading mmlu dataset.
    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1998/files
    if config.eval.tasks and "mmlu" in config.eval.tasks:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    # Evaluate the model
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=eval_wrapper,
            tasks=OmegaConf.to_container(config.eval.tasks),
            num_fewshot=config.eval.num_fewshot,
            limit=config.eval.limit,
        )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


def _convert_cli_to_config_format(args) -> DictConfig:
    """Convert CLI arguments to config format."""
    # First convert common args using the shared function
    config = _convert_args_to_config(args)

    # Add evaluation-specific settings
    config.eval = OmegaConf.create()
    config.eval.tasks = args.tasks
    config.eval.limit = args.limit
    config.eval.num_fewshot = args.num_fewshot
    config.eval.pte = args.pte
    config.eval.tokenizer_bin = args.tokenizer_bin
    config.eval.output_eager_checkpoint_file = args.output_eager_checkpoint_file
    config.eval.attention_sink_eval_tokens = args.attention_sink_eval_tokens

    return config


def build_args_parser() -> argparse.ArgumentParser:
    """Build argument parser for evaluation, extending the export parser with eval-specific args."""
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
        "--limit",
        type=int,
        default=None,
        help="number of samples to evalulate. If not set, evaluate all samples",
    )
    parser.add_argument(
        "-f",
        "--num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
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
    parser.add_argument(
        "--output_eager_checkpoint_file",
        type=str,
        default=None,
        help="Save the checkpoint after source transformations, for other evaluation platform to run the same checkpoint.",
    )

    # Set of parameters specific to AttentionSink.
    parser.add_argument("--attention_sink_eval_tokens", type=int, default=0)

    return parser
