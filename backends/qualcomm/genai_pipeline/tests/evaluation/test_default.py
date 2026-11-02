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
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.evaluation.default import (
    DefaultEvaluator,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    EvaluationScore,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteOutput,
    ExecuteRequest,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)


def _make_model():
    """Create a deployed model returning one pulled output file."""
    return MagicMock(return_value=ExecuteOutput(files=[Path("/tmp/out/output_0.raw")]))


def _make_input_config(**inference_options):
    """Create an InferenceInputConfig with the given inference options."""
    return InferenceInputConfig(
        soc_model=TEST_SOC_CHIPSET,
        artifact_dir=Path("/tmp/artifacts"),
        inference_options=inference_options,
    )


class TestDefaultEvaluator(unittest.TestCase):
    def test_method_is_default(self):
        """The evaluator reports under ``default``."""
        self.assertEqual(DefaultEvaluator.method, "default")

    def test_runs_all_samples_in_one_execute_request(self):
        """The samples run once with the default method index and iteration."""
        samples = [(1, 2), (3, 4)]
        model = _make_model()
        DefaultEvaluator(inputs=samples).evaluate(
            model, ControlArgs(), _make_input_config(), None
        )
        model.assert_called_once_with(
            ExecuteRequest(inputs=((1, 2), (3, 4)), method_index=0, iteration=1)
        )

    def test_reads_method_index_and_iteration_from_inference_options(self):
        """``method_index`` and ``iteration`` come from the inference options."""
        model = _make_model()
        DefaultEvaluator(inputs=[(1,)]).evaluate(
            model,
            ControlArgs(),
            _make_input_config(method_index=2, iteration=5),
            None,
        )
        (request,), _ = model.call_args
        self.assertEqual((request.method_index, request.iteration), (2, 5))

    def test_scores_nothing(self):
        """The runner's raw outputs are not scored."""
        score = DefaultEvaluator(inputs=[(1,)]).evaluate(
            _make_model(), ControlArgs(), _make_input_config(), None
        )
        self.assertEqual(score, EvaluationScore())

    def test_samples_are_copied_at_construction(self):
        """Mutating the caller's sample list does not change what runs."""
        samples = [(1,)]
        evaluator = DefaultEvaluator(inputs=samples)
        samples.append((2,))
        model = _make_model()
        evaluator.evaluate(model, ControlArgs(), _make_input_config(), None)
        (request,), _ = model.call_args
        self.assertEqual(request.inputs, ((1,),))


if __name__ == "__main__":
    unittest.main()
