# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator import (
    EvaluationScore,
    Evaluator,
)


class _CompleteEvaluator:
    """Evaluator stand-in exposing both protocol members."""

    method = "complete"

    def evaluate(self, model, control_args, input_config, metadata):
        """Return an empty score."""
        return EvaluationScore()


class _MissingEvaluate:
    """Evaluator stand-in without ``evaluate``."""

    method = "incomplete"


class TestEvaluationScore(unittest.TestCase):
    def test_defaults_are_empty(self):
        """A bare score has no generations and no metrics."""
        score = EvaluationScore()
        self.assertEqual(score.generations, [])
        self.assertEqual(score.metrics, {})

    def test_defaults_are_not_shared(self):
        """Each score gets its own default containers."""
        self.assertIsNot(EvaluationScore().metrics, EvaluationScore().metrics)
        self.assertIsNot(EvaluationScore().generations, EvaluationScore().generations)

    def test_is_frozen(self):
        """A score cannot be reassigned after the evaluator returns it."""
        score = EvaluationScore(metrics={"sqnr": 1.0})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            score.metrics = {}


class TestEvaluatorProtocol(unittest.TestCase):
    def test_accepts_class_with_method_and_evaluate(self):
        """A class with ``method`` and ``evaluate`` is an ``Evaluator``."""
        self.assertIsInstance(_CompleteEvaluator(), Evaluator)

    def test_rejects_class_without_evaluate(self):
        """A class missing ``evaluate`` is not an ``Evaluator``."""
        self.assertNotIsInstance(_MissingEvaluate(), Evaluator)


if __name__ == "__main__":
    unittest.main()
