# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    RunnerMetadata,
    RunnerOutput,
    RunnerRequest,
)

#: The deployed model as an evaluator sees it: runs one input on the deployed
#: runner and returns its prediction, typed after the input -- a
#: ``GenerateOutput`` for a ``GenerateRequest`` and a ``ForwardOutput`` for a
#: ``ForwardRequest``.
DeployedModel = Callable[[RunnerRequest], RunnerOutput]


@dataclass(frozen=True)
class EvaluationScore:
    """What one evaluation method concludes from the model's predictions.

    Attributes:
        generations: Generated texts the method reports, e.g. for
            ``prompt_eval``.
        metrics: Quality metrics keyed by name, e.g. ``sqnr`` or lm_eval's
            per-task results.
    """

    generations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """One evaluation method: builds the inputs fed to the deployed model and
    scores its predictions.

    ``model`` is the strategy's wrapper around the runner adapter and the
    evaluator's only way to the runner; the strategy records every call, so it
    owns the throughput and the result files.
    """

    method: str

    def evaluate(
        self,
        model: DeployedModel,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        metadata: Optional[RunnerMetadata],
    ) -> EvaluationScore:
        """Feed this method's inputs to ``model`` and score its predictions.

        Args:
            model: Runs one input on the deployed runner and returns its
                prediction.
            control_args: The run's control args, built once by the strategy,
                with the prompts and evaluation settings.
            input_config: Inference configuration with the tokenizer and the
                host model.
            metadata: The deployed decoder's metadata, or ``None`` when the
                adapter reads none.

        Returns:
            This method's generations and metrics.
        """
        ...
