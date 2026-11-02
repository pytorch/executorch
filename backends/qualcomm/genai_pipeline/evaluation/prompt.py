# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``prompt_eval``: generate text for the pipeline prompts."""

from __future__ import annotations

from typing import Optional

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    PROMPT_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    DeployedModel,
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    GenerateRequest,
    RunnerMetadata,
)


class PromptEvaluator:
    """Generate text for the pipeline prompts and report the generations.

    Feeds one ``GenerateRequest`` over the pipeline prompts.
    """

    method = PROMPT_EVAL

    def evaluate(
        self,
        model: DeployedModel,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        metadata: Optional[RunnerMetadata],
    ) -> EvaluationScore:
        """Generate one response over the pipeline prompts.

        Args:
            model: Runs the generation on the deployed runner.
            control_args: Control args that supply the prompts.
            input_config: Unused; part of the evaluator interface.
            metadata: Unused; generation needs no decoder metadata.

        Returns:
            The generated text.
        """
        prediction = model(GenerateRequest(prompts=list(control_args.prompt)))
        return EvaluationScore(generations=prediction.texts)
