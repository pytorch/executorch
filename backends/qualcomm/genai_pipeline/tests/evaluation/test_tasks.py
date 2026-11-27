# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    TASKS_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.tasks import LmEvalEvaluator
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardOutput,
    ForwardRequest,
    RunnerMetadata,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper

_EAGER_EVAL_INIT = (
    "executorch.examples.models.llama.evaluate.eager_eval.EagerEvalWrapper.__init__"
)
_SIMPLE_EVALUATE = "lm_eval.evaluator.simple_evaluate"
_MAX_CONTEXT_LEN = 16
_LOGITS = torch.ones(1, 3, 4)


def _fake_eager_eval_init(self, model, tokenizer, max_seq_length=None, **kwargs):
    """Stand in for ``EagerEvalWrapper.__init__`` without loading a HF model."""
    self._model = model
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length


def _make_tokenizer_wrapper():
    """Create a ``TokenizerWrapper`` stand-in carrying a bare tokenizer."""
    wrapper = MagicMock(spec=TokenizerWrapper)
    wrapper.tokenizer = MagicMock(name="tokenizer")
    return wrapper


def _make_metadata():
    """Create decoder metadata with a small context length."""
    return RunnerMetadata(
        max_context_len=_MAX_CONTEXT_LEN,
        max_seq_len=_MAX_CONTEXT_LEN,
        vocab_size=4,
        logits_scale=1.0,
        logits_zero_point=0,
        kv_io_bit_width=16,
    )


def _make_input_config(tokenizer):
    """Create an InferenceInputConfig carrying ``tokenizer``."""
    return InferenceInputConfig(
        soc_model=TEST_SOC_CHIPSET,
        artifact_dir=Path("/tmp/artifacts"),
        tokenizer=tokenizer,
    )


def _make_control_args(**overrides):
    """Create control args requesting one lm_eval task."""
    defaults = {"eval_tasks": ["wikitext"], "eval_limit": 2, "eval_num_fewshot": 0}
    defaults.update(overrides)
    return ControlArgs(**defaults)


def _make_model():
    """Create a deployed model returning fixed logits."""
    return MagicMock(return_value=ForwardOutput(logits=_LOGITS, tokens_per_sec=1.0))


@patch(_EAGER_EVAL_INIT, _fake_eager_eval_init)
class TestLmEvalEvaluator(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _make_tokenizer_wrapper()
        self.input_config = _make_input_config(self.tokenizer)

    def test_method_is_tasks_eval(self):
        """The evaluator reports under ``tasks_eval``."""
        self.assertEqual(LmEvalEvaluator.method, TASKS_EVAL)

    def test_forwards_eval_settings_and_returns_results(self):
        """lm_eval gets the tasks, few-shot count, and limit; its results are the metrics."""
        results = {"wikitext": {"word_perplexity,none": 12.5}}
        with patch(_SIMPLE_EVALUATE, return_value={"results": results}) as evaluate:
            score = LmEvalEvaluator().evaluate(
                _make_model(), _make_control_args(), self.input_config, _make_metadata()
            )
        _, kwargs = evaluate.call_args
        self.assertEqual(kwargs["tasks"], ["wikitext"])
        self.assertEqual(kwargs["num_fewshot"], 0)
        self.assertEqual(kwargs["limit"], 2)
        self.assertEqual(score.metrics, results)
        self.assertEqual(score.generations, [])

    def test_lm_eval_model_wraps_the_bare_tokenizer_and_context(self):
        """The lm_eval model uses the bare tokenizer and a context minus one."""
        with patch(_SIMPLE_EVALUATE, return_value={"results": {}}) as evaluate:
            LmEvalEvaluator().evaluate(
                _make_model(), _make_control_args(), self.input_config, _make_metadata()
            )
        lm_model = evaluate.call_args.kwargs["model"]
        self.assertIs(lm_model._tokenizer, self.tokenizer.tokenizer)
        self.assertIsNone(lm_model._model)
        self.assertEqual(lm_model._max_seq_length, _MAX_CONTEXT_LEN - 1)

    def test_model_call_runs_one_forward_request_over_its_tokens(self):
        """Each lm_eval model call is one ``ForwardRequest`` with no extra decode."""
        model = _make_model()
        with patch(_SIMPLE_EVALUATE, return_value={"results": {}}) as evaluate:
            LmEvalEvaluator().evaluate(
                model, _make_control_args(), self.input_config, _make_metadata()
            )
        lm_model = evaluate.call_args.kwargs["model"]
        logits = lm_model._model_call(torch.tensor([[5, 6, 7]]))
        model.assert_called_once_with(ForwardRequest(token_ids=[5, 6, 7]))
        self.assertIs(logits, _LOGITS)

    def test_model_call_rejects_batched_input(self):
        """A batched lm_eval model call is rejected."""
        with patch(_SIMPLE_EVALUATE, return_value={"results": {}}) as evaluate:
            LmEvalEvaluator().evaluate(
                _make_model(), _make_control_args(), self.input_config, _make_metadata()
            )
        lm_model = evaluate.call_args.kwargs["model"]
        for inps in (torch.zeros(2, 3, dtype=torch.long), torch.zeros(3)):
            with self.subTest(shape=tuple(inps.shape)):
                with self.assertRaisesRegex(ValueError, r"\[1, sequence_length\]"):
                    lm_model._model_call(inps)

    def test_requires_eval_tasks(self):
        """Missing tasks are rejected before lm_eval runs."""
        with patch(_SIMPLE_EVALUATE) as evaluate:
            with self.assertRaisesRegex(ValueError, "eval_tasks is required"):
                LmEvalEvaluator().evaluate(
                    _make_model(),
                    _make_control_args(eval_tasks=None),
                    self.input_config,
                    _make_metadata(),
                )
        evaluate.assert_not_called()

    def test_requires_tokenizer_wrapper(self):
        """A tokenizer that is not a ``TokenizerWrapper`` is rejected."""
        for tokenizer in (None, MagicMock(name="bare_tokenizer")):
            with self.subTest(tokenizer=tokenizer):
                with self.assertRaisesRegex(ValueError, "requires a tokenizer"):
                    LmEvalEvaluator().evaluate(
                        _make_model(),
                        _make_control_args(),
                        _make_input_config(tokenizer),
                        _make_metadata(),
                    )

    def test_requires_wrapped_tokenizer(self):
        """A ``TokenizerWrapper`` without a loaded tokenizer is rejected."""
        self.tokenizer.tokenizer = None
        with self.assertRaisesRegex(ValueError, "requires a tokenizer"):
            LmEvalEvaluator().evaluate(
                _make_model(), _make_control_args(), self.input_config, _make_metadata()
            )

    def test_requires_decoder_metadata(self):
        """Missing decoder metadata is rejected."""
        with self.assertRaisesRegex(ValueError, "decoder's metadata"):
            LmEvalEvaluator().evaluate(
                _make_model(), _make_control_args(), self.input_config, None
            )


if __name__ == "__main__":
    unittest.main()
