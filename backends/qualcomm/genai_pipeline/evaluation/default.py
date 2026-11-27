# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``default``: run a compiled method through the generic executor runner."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    DEFAULT_RUN,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    DeployedModel,
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteRequest,
    RunnerMetadata,
)


class DefaultEvaluator:
    """Run one compiled method on caller-supplied data samples.

    Pairs with ``DefaultDeviceRunnerAdapter``. The samples are the real data to
    run -- as the legacy scripts load a dataset before pushing it -- not model
    preparation's example inputs, which only describe the export signature. The
    runner writes raw output tensors whose layout only the caller knows, so
    nothing is scored: the pulled files reach the method's result as they are.

    Args:
        inputs: The samples to run, one positional-input tuple each.
        method_index: The compiled method to run.
        iteration: How many times the runner runs it over the samples.
    """

    method = DEFAULT_RUN

    def __init__(
        self,
        inputs: Sequence[Tuple[Any, ...]],
        method_index: int = 0,
        iteration: int = 1,
    ) -> None:
        self._inputs = tuple(inputs)
        self._method_index = method_index
        self._iteration = iteration

    def evaluate(
        self,
        model: DeployedModel,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        metadata: Optional[RunnerMetadata],
    ) -> EvaluationScore:
        """Run the compiled method once over the samples.

        Args:
            model: Runs the method on the deployed runner.
            control_args: Unused; part of the evaluator interface.
            input_config: Unused; part of the evaluator interface.
            metadata: Unused; the generic runner has no decoder metadata.

        Returns:
            An empty score; the runner's output files reach the method's result
            through its prediction.
        """
        model(
            ExecuteRequest(
                inputs=self._inputs,
                method_index=self._method_index,
                iteration=self._iteration,
            )
        )
        return EvaluationScore()
