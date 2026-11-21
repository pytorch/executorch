# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``tasks_eval``: lm_eval tasks scored through runner logits."""

from __future__ import annotations

from typing import Any, Optional

import torch

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    TASKS_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    DeployedModel,
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardRequest,
    RunnerMetadata,
)


class LmEvalEvaluator:
    """Run lm_eval tasks, answering each of its model calls on the deployed runner.

    lm_eval asks for logits from inside ``simple_evaluate``; each of those model
    calls runs as one ``ForwardRequest`` on the deployed model.
    """

    method = TASKS_EVAL

    def evaluate(
        self,
        model: DeployedModel,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        metadata: Optional[RunnerMetadata],
    ) -> EvaluationScore:
        """Run lm_eval against the deployed model and collect its task metrics.

        Args:
            model: Runs each of lm_eval's model calls on the deployed runner.
            control_args: Control args that supply the tasks, limit, and
                few-shot count.
            input_config: Inference configuration that supplies the tokenizer.
            metadata: The deployed decoder's metadata, which supplies its
                context length.

        Returns:
            lm_eval's ``results``, keyed by task.

        Raises:
            ValueError: If no tasks, tokenizer, or decoder metadata are
                available, or if lm_eval issues a batched model call.
        """
        from executorch.examples.models.llama.evaluate.eager_eval import (
            EagerEvalWrapper,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
            TokenizerWrapper,
        )
        from lm_eval.evaluator import simple_evaluate

        if not control_args.eval_tasks:
            raise ValueError("eval_tasks is required for tasks_eval")
        wrapper = input_config.tokenizer
        if not isinstance(wrapper, TokenizerWrapper) or wrapper.tokenizer is None:
            raise ValueError("tasks_eval requires a tokenizer from model preparation")
        if metadata is None:
            raise ValueError("tasks_eval requires the deployed decoder's metadata")

        # Defined here because EagerEvalWrapper lives under examples/, which this
        # package imports lazily.
        class _QnnRunnerEvalWrapper(EagerEvalWrapper):
            """lm_eval model whose logits come from the deployed runner.

            Args:
                deployed_model: Runs one input on the deployed runner.
                tokenizer: The bare tokenizer lm_eval encodes with.
                max_context_len: The deployed decoder's context length.
            """

            def __init__(
                self,
                deployed_model: DeployedModel,
                tokenizer: Any,
                max_context_len: int,
            ) -> None:
                # No host module: the logits come from the runner.
                super().__init__(None, tokenizer, max_context_len - 1)
                # Not ``_model``, which EagerEvalWrapper already sets.
                self._deployed_model = deployed_model
                self._max_context_len = max_context_len

            def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
                if inps.ndim != 2 or inps.shape[0] != 1:
                    raise ValueError(
                        "tasks_eval requires token IDs shaped [1, sequence_length]"
                    )
                # Decode to the full context, as tasks_eval always has.
                model_input = ForwardRequest(
                    token_ids=inps[0].tolist(), seq_len=self._max_context_len
                )
                return self._deployed_model(model_input).logits

        with torch.no_grad():
            results = simple_evaluate(
                model=_QnnRunnerEvalWrapper(
                    model, wrapper.tokenizer, metadata.max_context_len
                ),
                tasks=control_args.eval_tasks,
                num_fewshot=control_args.eval_num_fewshot,
                limit=control_args.eval_limit,
            )
        return EvaluationScore(metrics=results["results"])
