# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Union

import torch
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.qualcomm.oss_scripts.llama.inference import DecoderInference
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )


class GraphModuleCalibrationWrapper(EagerEvalWrapper):
    """Wraps a GraphModule for lm_eval-based evaluation using DecoderInference."""

    def __init__(
        self,
        model: torch.fx.GraphModule,
        tokenizer: Union[
            SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
        ],
        max_seq_length: int,
        get_example_inputs: Callable,
        use_i64_token: bool,
    ):
        assert max_seq_length is not None, "max_seq_length must be provided"
        super().__init__(
            model=model, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
        self._model = model.to(self.device)
        self._runner = DecoderInference(
            get_example_inputs=get_example_inputs,
            max_context_len=max_seq_length,
            use_i64_token=use_i64_token,
        )

    def _model_call(self, inps):
        logits = self._runner.predict_step(
            self._model,
            input_ids=inps,
        )
        return logits


def run_lm_eval(
    module: torch.fx.GraphModule,
    get_example_inputs: Callable,
    tokenizer: Union[SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer],
    max_seq_length: int,
    tasks,
    use_i64_token: bool = False,
    num_fewshot: Optional[int] = None,
    limit: int = 1,
    max_batch_size: int = 1,
    event_name: str = "",
) -> None:
    assert tasks, "tasks must be a non-empty list of lm_eval task names"

    wrapper = GraphModuleCalibrationWrapper(
        model=module,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        get_example_inputs=get_example_inputs,
        use_i64_token=use_i64_token,
    )
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=wrapper,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=max_batch_size,
        )
    logging.info("Evaluation summary for %s", event_name)
    for task, res in eval_results["results"].items():
        logging.info("%s: %s", task, res)
