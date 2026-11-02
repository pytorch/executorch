# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.control_args import (
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import Evaluator
from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator_lookup import (
    get_evaluators,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.prompt import (
    PromptEvaluator,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.sqnr import SqnrEvaluator
from executorch.backends.qualcomm.genai_pipeline.evaluation.tasks import LmEvalEvaluator


class TestGetEvaluators(unittest.TestCase):
    def test_maps_each_text_method_to_its_evaluator(self):
        """Every text-model method resolves to its evaluator class."""
        evaluators = get_evaluators(
            [PROMPT_EVAL, TASKS_EVAL, SQNR_EVAL], multimodal=False
        )
        self.assertEqual(
            [type(evaluator) for evaluator in evaluators],
            [PromptEvaluator, LmEvalEvaluator, SqnrEvaluator],
        )

    def test_preserves_requested_order(self):
        """Evaluators come back in the order the methods were requested."""
        evaluators = get_evaluators([SQNR_EVAL, PROMPT_EVAL], multimodal=False)
        self.assertEqual(
            [evaluator.method for evaluator in evaluators], [SQNR_EVAL, PROMPT_EVAL]
        )

    def test_every_evaluator_satisfies_the_protocol(self):
        """Each built evaluator is an ``Evaluator`` named after its method."""
        for method in (PROMPT_EVAL, TASKS_EVAL, SQNR_EVAL):
            with self.subTest(method=method):
                (evaluator,) = get_evaluators([method], multimodal=False)
                self.assertIsInstance(evaluator, Evaluator)
                self.assertEqual(evaluator.method, method)

    def test_multimodal_supports_prompt_eval(self):
        """A multimodal model can run ``prompt_eval``."""
        (evaluator,) = get_evaluators([PROMPT_EVAL], multimodal=True)
        self.assertIsInstance(evaluator, PromptEvaluator)

    def test_multimodal_rejects_logit_methods(self):
        """A multimodal model rejects the logit-based methods by name."""
        for method in (TASKS_EVAL, SQNR_EVAL):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, f"{method}.*multimodal"):
                    get_evaluators([PROMPT_EVAL, method], multimodal=True)

    def test_rejects_unknown_method(self):
        """An unknown method is rejected for text models too."""
        with self.assertRaisesRegex(ValueError, "unknown_eval unsupported for text"):
            get_evaluators(["unknown_eval"], multimodal=False)

    def test_empty_request_builds_no_evaluators(self):
        """No requested method yields no evaluators."""
        self.assertEqual(get_evaluators([], multimodal=False), [])


if __name__ == "__main__":
    unittest.main()
