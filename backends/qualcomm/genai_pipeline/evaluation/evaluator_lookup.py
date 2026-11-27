# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Map evaluation-method names to the evaluators that implement them."""

from __future__ import annotations

from typing import Dict, List, Sequence, Type

from executorch.backends.qualcomm.genai_pipeline.control_args import (
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import Evaluator
from executorch.backends.qualcomm.genai_pipeline.evaluation.prompt import (
    PromptEvaluator,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.sqnr import SqnrEvaluator
from executorch.backends.qualcomm.genai_pipeline.evaluation.tasks import LmEvalEvaluator

_TEXT_EVALUATORS: Dict[str, Type[Evaluator]] = {
    PROMPT_EVAL: PromptEvaluator,
    TASKS_EVAL: LmEvalEvaluator,
    SQNR_EVAL: SqnrEvaluator,
}
_MULTIMODAL_EVALUATORS: Dict[str, Type[Evaluator]] = {
    PROMPT_EVAL: PromptEvaluator,
}


def get_evaluators(eval_methods: Sequence[str], multimodal: bool) -> List[Evaluator]:
    """Build one evaluator per requested evaluation method, in order.

    Args:
        eval_methods: Evaluation-method names, such as ``prompt_eval``.
        multimodal: Whether the model is multimodal, which supports only
            ``prompt_eval``.

    Returns:
        The evaluators for ``eval_methods``.

    Raises:
        ValueError: If a method is unsupported for the model kind.
    """
    evaluator_types = _MULTIMODAL_EVALUATORS if multimodal else _TEXT_EVALUATORS
    unsupported = [method for method in eval_methods if method not in evaluator_types]
    if unsupported:
        model_kind = "multimodal" if multimodal else "text"
        raise ValueError(
            f"{', '.join(unsupported)} unsupported for {model_kind} models"
        )
    return [evaluator_types[eval_method]() for eval_method in eval_methods]
