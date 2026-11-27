# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    DECODE_QDQ_FILENAME,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    SQNR_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.evaluation.sqnr import (
    build_host_reference,
    build_prompt_token_ids,
    forward_prompt,
    HostReference,
    score_sqnr,
    SqnrEvaluator,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardOutput,
    ForwardRequest,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper

_SQNR = "executorch.backends.qualcomm.genai_pipeline.evaluation.sqnr"
_DECODER_INFERENCE = (
    "executorch.examples.qualcomm.oss_scripts.llama.inference.DecoderInference"
)
_LLM_COLLECTOR = (
    "executorch.backends.qualcomm.genai_pipeline.datasets.collators.llm_collator."
    "LLMDatasetCollector"
)
_MAX_CONTEXT_LEN = 8


def _make_input_config(artifact_dir, **overrides):
    """Create an InferenceInputConfig carrying a decoder and its example inputs."""
    defaults = {
        "soc_model": TEST_SOC_CHIPSET,
        "artifact_dir": Path(artifact_dir),
        "model_module": {ARTIFACT_TEXT_DECODER: torch.nn.Identity()},
        "example_inputs": {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_FORWARD: (torch.zeros(1, _MAX_CONTEXT_LEN),),
                GRAPH_KV_FORWARD: (torch.zeros(1, 1),),
            }
        },
    }
    defaults.update(overrides)
    return InferenceInputConfig(**defaults)


def _make_reference(qdq_path):
    """Create a HostReference whose QDQ graph lives at ``qdq_path``."""
    return HostReference(
        fp32_module=MagicMock(name="fp32_module"),
        decoder_inputs=(torch.zeros(1, _MAX_CONTEXT_LEN),),
        max_context_len=_MAX_CONTEXT_LEN,
        use_i64_token=False,
        qdq_path=Path(qdq_path),
    )


def _make_tokenizer_wrapper():
    """Create a ``TokenizerWrapper`` stand-in with a chat template."""
    wrapper = MagicMock(spec=TokenizerWrapper)
    wrapper.make_chat_template.return_value = [{"role": "user", "content": "hi"}]
    wrapper.apply_chat_template.return_value = "<user>hi"
    wrapper.tokenizer = MagicMock(name="tokenizer")
    wrapper.tokenizer.encode.return_value = [1, 2, 3]
    return wrapper


class TestBuildHostReference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_selects_decoder_calibration_graph(self):
        """The reference uses the decoder module and its calibration-graph inputs."""
        input_config = _make_input_config(self.tmp.name)
        reference = build_host_reference(
            ControlArgs(max_context_len=_MAX_CONTEXT_LEN), input_config
        )
        self.assertIs(
            reference.fp32_module, input_config.model_module[ARTIFACT_TEXT_DECODER]
        )
        self.assertIs(
            reference.decoder_inputs,
            input_config.example_inputs[ARTIFACT_TEXT_DECODER][GRAPH_FORWARD],
        )
        self.assertEqual(reference.max_context_len, _MAX_CONTEXT_LEN)

    def test_use_i64_token_follows_embedding_quantize(self):
        """int64 tokens are used exactly when the embedding is quantized."""
        input_config = _make_input_config(self.tmp.name)
        for embedding_quantize, expected in ((None, False), ("4,32", True)):
            with self.subTest(embedding_quantize=embedding_quantize):
                reference = build_host_reference(
                    ControlArgs(embedding_quantize=embedding_quantize), input_config
                )
                self.assertEqual(reference.use_i64_token, expected)

    def test_qdq_path_defaults_to_artifact_dir(self):
        """Without ``--pre-gen-pte``, the QDQ graph is read from the artifact dir."""
        reference = build_host_reference(
            ControlArgs(), _make_input_config(self.tmp.name)
        )
        self.assertEqual(reference.qdq_path, Path(self.tmp.name) / DECODE_QDQ_FILENAME)

    def test_qdq_path_prefers_pre_gen_pte(self):
        """With ``--pre-gen-pte``, the QDQ graph is read from that directory."""
        reference = build_host_reference(
            ControlArgs(pre_gen_pte="/tmp/pre_gen"), _make_input_config(self.tmp.name)
        )
        self.assertEqual(reference.qdq_path, Path("/tmp/pre_gen") / DECODE_QDQ_FILENAME)

    def test_requires_model_preparation_outputs(self):
        """A missing FP32 module or example inputs is rejected."""
        for missing in ("model_module", "example_inputs"):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, "FP32 module"):
                    build_host_reference(
                        ControlArgs(),
                        _make_input_config(self.tmp.name, **{missing: None}),
                    )


class TestBuildPromptTokenIds(unittest.TestCase):
    def test_tokenizes_chat_templated_prompt_with_bos_only(self):
        """The prompt is chat-templated, then encoded with BOS and without EOS."""
        wrapper = _make_tokenizer_wrapper()
        token_ids = build_prompt_token_ids(wrapper, "hi", "be brief")
        wrapper.make_chat_template.assert_called_once_with("hi", "be brief")
        wrapper.apply_chat_template.assert_called_once_with(
            wrapper.make_chat_template.return_value
        )
        wrapper.tokenizer.encode.assert_called_once_with(
            "<user>hi", bos=True, eos=False
        )
        self.assertEqual(token_ids, [1, 2, 3])

    def test_rejects_non_tokenizer_wrapper(self):
        """A tokenizer without chat-template support is rejected by type name."""
        with self.assertRaisesRegex(TypeError, "got MagicMock"):
            build_prompt_token_ids(MagicMock(), "hi", "")


class TestForwardPrompt(unittest.TestCase):
    def test_returns_prompt_position_logits(self):
        """The full-sequence logits are cut to the prompt positions."""
        decoder_inputs = (torch.zeros(1, _MAX_CONTEXT_LEN),)
        batch = {"input_ids": MagicMock(name="ids"), "attention_mask": MagicMock()}
        collator = MagicMock(return_value=batch)
        full_logits = torch.arange(_MAX_CONTEXT_LEN * 2, dtype=torch.float32).reshape(
            1, _MAX_CONTEXT_LEN, 2
        )
        module = MagicMock(name="module")
        with patch(_LLM_COLLECTOR) as collector, patch(
            _DECODER_INFERENCE
        ) as decoder_cls:
            collector.return_value.create_collators.return_value = {
                ARTIFACT_TEXT_DECODER: collator
            }
            decoder_cls.return_value.predict_step.return_value = full_logits
            logits = forward_prompt(
                module, decoder_inputs, [7, 8, 9], _MAX_CONTEXT_LEN, True
            )

        collator.assert_called_once_with([([7, 8, 9],)])
        _, decoder_kwargs = decoder_cls.call_args
        self.assertEqual(decoder_kwargs["max_context_len"], _MAX_CONTEXT_LEN)
        self.assertTrue(decoder_kwargs["use_i64_token"])
        self.assertIs(decoder_kwargs["get_example_inputs"](), decoder_inputs)
        decoder_cls.return_value.predict_step.assert_called_once_with(
            module, batch["input_ids"], attn_mask=batch["attention_mask"]
        )
        self.assertTrue(torch.equal(logits, full_logits[:, :3, :]))


class TestScoreSqnr(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.qnn_logits = torch.tensor([[[1.0, 2.0, 3.0]]])

    def test_skips_qdq_comparison_without_saved_graph(self):
        """Without a saved QDQ graph only the FP32 SQNR is reported."""
        reference = _make_reference(Path(self.tmp.name) / DECODE_QDQ_FILENAME)
        with patch(f"{_SQNR}.forward_prompt", return_value=self.qnn_logits) as forward:
            sqnr, qdq_sqnr = score_sqnr(reference, [1], self.qnn_logits)
        forward.assert_called_once_with(
            reference.fp32_module,
            reference.decoder_inputs,
            [1],
            _MAX_CONTEXT_LEN,
            False,
        )
        self.assertEqual(sqnr, float("inf"))
        self.assertIsNone(qdq_sqnr)

    def test_scores_against_saved_qdq_graph(self):
        """Each SQNR uses its own golden as the reference signal."""
        qdq_path = Path(self.tmp.name) / DECODE_QDQ_FILENAME
        qdq_path.touch()
        reference = _make_reference(qdq_path)
        # The errors against QNN are 1.0 throughout, so each SQNR is
        # 20 * log10(|golden|): swapping compute_error's arguments, or scoring
        # another tensor, changes the value.
        fp32_logits = torch.tensor([[[10.0, 0.0]]])
        qdq_logits = torch.tensor([[[8.0, 0.0]]])
        qnn_logits = torch.tensor([[[9.0, 0.0]]])
        qdq_module = MagicMock(name="qdq_module")
        with patch(
            f"{_SQNR}.forward_prompt", side_effect=[fp32_logits, qdq_logits]
        ) as forward, patch("torch.export.load") as load:
            load.return_value.module.return_value = qdq_module
            sqnr, qdq_sqnr = score_sqnr(reference, [1], qnn_logits)

        load.assert_called_once_with(qdq_path)
        self.assertIs(forward.call_args_list[0].args[0], reference.fp32_module)
        self.assertIs(forward.call_args_list[1].args[0], qdq_module)
        self.assertAlmostEqual(sqnr, 20.0, places=4)
        self.assertAlmostEqual(qdq_sqnr, 20 * math.log10(8.0), places=4)


class TestSqnrEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logits = torch.zeros(1, 3, 4)
        self.model = MagicMock(
            return_value=ForwardOutput(logits=self.logits, tokens_per_sec=1.0)
        )
        self.control_args = ControlArgs(
            prompt=["hi"], system_prompt="be brief", max_context_len=_MAX_CONTEXT_LEN
        )
        self.input_config = _make_input_config(
            self.tmp.name, tokenizer=MagicMock(name="tokenizer")
        )

    def _evaluate(self, scores):
        """Run the evaluator with tokenization and scoring patched."""
        with patch(
            f"{_SQNR}.build_prompt_token_ids", return_value=[1, 2, 3]
        ) as tokenize, patch(f"{_SQNR}.score_sqnr", return_value=scores) as score:
            result = SqnrEvaluator().evaluate(
                self.model, self.control_args, self.input_config, None
            )
        return result, tokenize, score

    def test_method_is_sqnr_eval(self):
        """The evaluator reports under ``sqnr_eval``."""
        self.assertEqual(SqnrEvaluator.method, SQNR_EVAL)

    def test_forwards_first_prompt_and_scores_its_logits(self):
        """The first prompt is tokenized, forwarded once, and its logits scored."""
        _, tokenize, score = self._evaluate((20.0, None))
        tokenize.assert_called_once_with(self.input_config.tokenizer, "hi", "be brief")
        self.model.assert_called_once_with(ForwardRequest(token_ids=[1, 2, 3]))
        reference, token_ids, logits = score.call_args.args
        self.assertIsInstance(reference, HostReference)
        self.assertEqual(token_ids, [1, 2, 3])
        self.assertIs(logits, self.logits)

    def test_reports_sqnr_only_without_qdq_graph(self):
        """Only ``sqnr`` is reported when the QDQ graph is missing."""
        score, _, _ = self._evaluate((20.0, None))
        self.assertEqual(score.metrics, {"sqnr": 20.0})
        self.assertEqual(score.generations, [])

    def test_reports_qdq_sqnr_when_available(self):
        """``qdq_sqnr`` is reported alongside ``sqnr`` when scored."""
        score, _, _ = self._evaluate((20.0, 25.0))
        self.assertEqual(score.metrics, {"sqnr": 20.0, "qdq_sqnr": 25.0})

    def test_requires_model_preparation_outputs(self):
        """Without the FP32 module the runner is never called."""
        self.input_config = _make_input_config(self.tmp.name, model_module=None)
        with self.assertRaisesRegex(ValueError, "FP32 module"):
            self._evaluate((20.0, None))
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
