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
    """One evaluation method: feeds inputs to the deployed model and scores its
    predictions.

    An evaluator owns two things, and only these:

    - **What to feed the model.** It builds each input -- a ``GenerateRequest``
      over the prompts, a ``ForwardRequest`` over chat-templated tokens, the
      model calls lm_eval makes -- possibly from earlier predictions.
    - **How to score the predictions.** It turns what comes back into an
      ``EvaluationScore``: the generated text, or quality metrics such as SQNR
      or lm_eval's per-task results.

    It is the part of the inference stage that varies by evaluation method. The
    stage splits the rest by concern, and the evaluator reaches none of it
    directly::

        ExecuTorchInferenceStrategy   when: deploys the artifacts once, runs every
                                      evaluator, measures throughput, writes
                                      results.json / performance.json
        Evaluator                     what to feed and how to score (this class)
        runner adapter                how to run: runner flags, input staging,
                                      output decoding, the deployed model's
                                      metadata
        RunnerPlatform                where to run: device or x86 host

    The strategy wraps the adapter as ``model``, a ``DeployedModel`` that runs
    one input on the deployed runner and returns its prediction; that is the
    evaluator's only way to the runner. What the evaluator needs of the deployed
    model arrives as ``RunnerMetadata``, so it never reads the PTE. Every call to
    ``model`` is recorded by the strategy, which therefore owns the throughput
    and the result files; the evaluator neither measures one nor writes the
    other.

    Not to be confused with ``datasets.evaluation``, which supplies data for the
    quantization stage's post-quantization evaluation.
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
