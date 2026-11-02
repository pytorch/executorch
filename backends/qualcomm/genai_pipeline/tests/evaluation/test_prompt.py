# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    PROMPT_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.prompt import (
    PromptEvaluator,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    GenerateOutput,
    GenerateRequest,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)


def _make_model(texts):
    """Create a deployed model returning ``texts`` for every generation."""
    return MagicMock(
        return_value=GenerateOutput(texts=texts, tokens_per_sec=10.0),
    )


def _make_input_config():
    """Create a minimal InferenceInputConfig."""
    return InferenceInputConfig(
        soc_model=TEST_SOC_CHIPSET, artifact_dir=Path("/tmp/artifacts")
    )


class TestPromptEvaluator(unittest.TestCase):
    def test_method_is_prompt_eval(self):
        """The evaluator reports under ``prompt_eval``."""
        self.assertEqual(PromptEvaluator.method, PROMPT_EVAL)

    def test_feeds_one_generate_request_over_all_prompts(self):
        """All prompts go to the model in a single ``GenerateRequest``."""
        model = _make_model(["a", "b"])
        PromptEvaluator().evaluate(
            model,
            ControlArgs(prompt=["first", "second"]),
            _make_input_config(),
            None,
        )
        model.assert_called_once_with(GenerateRequest(prompts=["first", "second"]))

    def test_reports_generations_without_metrics(self):
        """The score carries the generated texts and no metrics."""
        score = PromptEvaluator().evaluate(
            _make_model(["hello"]),
            ControlArgs(prompt=["hi"]),
            _make_input_config(),
            None,
        )
        self.assertEqual(score, EvaluationScore(generations=["hello"]))

    def test_request_prompts_are_a_copy(self):
        """The request does not alias the control args' prompt list."""
        control_args = ControlArgs(prompt=["hi"])
        model = _make_model(["hello"])
        PromptEvaluator().evaluate(model, control_args, _make_input_config(), None)
        (request,), _ = model.call_args
        self.assertIsNot(request.prompts, control_args.prompt)


if __name__ == "__main__":
    unittest.main()
