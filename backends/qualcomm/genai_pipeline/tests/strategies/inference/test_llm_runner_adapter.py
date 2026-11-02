# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteRequest,
    ForwardOutput,
    ForwardRequest,
    GenerateOutput,
    GenerateRequest,
    RunnerMetadata,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    RUNNER_INFERENCE_SPEED,
    RUNNER_INPUT_TOKENS,
    RUNNER_LOGITS,
    RUNNER_OUTPUTS,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.llm_runner_adapter import (
    LLMRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_SOC_CHIPSET,
)

_READ_METADATA = (
    "executorch.backends.qualcomm.genai_pipeline.strategies.inference."
    "llm_runner_adapter.retrieve_info_from_pte"
)
_METADATA = RunnerMetadata(
    max_context_len=8,
    max_seq_len=8,
    vocab_size=4,
    logits_scale=0.5,
    logits_zero_point=2,
    kv_io_bit_width=16,
)


def _make_platform(runner_dir):
    """Create a mock RunnerPlatform whose runner outputs live in ``runner_dir``."""
    platform = MagicMock()
    platform.target_flags = ()
    platform.limits_generation = False
    platform.output_dir.return_value = "/out"
    platform.runner_path.side_effect = lambda path: Path(path).name
    platform.pull.side_effect = lambda filename, host_path: shutil.copy(
        runner_dir / filename, host_path
    )
    return platform


class TestLLMRunnerAdapter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.artifact_dir = self.tmp_dir / "artifacts"
        self.runner_dir = self.tmp_dir / "runner"
        self.runner_dir.mkdir()
        self.pte_path = self.tmp_dir / "stories110m_text_decoder.pte"
        self.pte_path.write_bytes(b"pte")
        self.tokenizer_path = self.tmp_dir / "tokenizer.bin"
        self.tokenizer_path.write_bytes(b"tok")
        self.platform = _make_platform(self.runner_dir)
        self.adapter = LLMRunnerAdapter(self.platform)
        self.context = make_test_context(
            model_name="stories110m",
            extra_options={"model_mode": "kv", "max_seq_len": 64},
        )
        patcher = patch(_READ_METADATA, return_value=_METADATA)
        self.mock_read_metadata = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _input_config(self, **overrides):
        """Create an InferenceInputConfig holding the decoder and tokenizer."""
        defaults = {
            "soc_model": TEST_SOC_CHIPSET,
            "artifact_dir": self.artifact_dir,
            "pte_paths": {ARTIFACT_TEXT_DECODER: self.pte_path},
            "runtime_tokenizer_path": self.tokenizer_path,
        }
        defaults.update(overrides)
        return InferenceInputConfig(**defaults)

    def _push_artifacts(self, **overrides):
        """Deploy the adapter with the default input configuration."""
        self.adapter.push_artifacts(self.context, self._input_config(**overrides))

    def _executed_args(self):
        """Return the flags of the last platform.execute call."""
        return self.platform.execute.call_args.args[0]

    def test_push_artifacts_sets_up_decoder_and_tokenizer(self):
        """The decoder PTE, extra files, and tokenizer are handed to the platform."""
        self.adapter.push_artifacts(
            self.context, self._input_config(), extra_files=["/host/extra.bin"]
        )
        self.platform.setup.assert_called_once_with(
            LLMRunnerAdapter._RUNNER,
            [self.pte_path, Path("/host/extra.bin"), self.tokenizer_path],
            self.artifact_dir / "outputs",
        )
        self.assertTrue((self.artifact_dir / "outputs").is_dir())

    def test_push_artifacts_requires_runtime_tokenizer(self):
        """Deploying without a runtime tokenizer raises ValueError."""
        with self.assertRaisesRegex(ValueError, "runtime tokenizer"):
            self._push_artifacts(runtime_tokenizer_path=None)

    def test_push_artifacts_requires_text_decoder(self):
        """Deploying without a text-decoder PTE raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Missing PTE artifact"):
            self._push_artifacts(pte_paths={})

    def test_get_metadata_reads_the_decoder_pte(self):
        """Metadata comes from the deployed decoder PTE."""
        self._push_artifacts()
        self.assertIs(self.adapter.get_metadata(), _METADATA)
        self.mock_read_metadata.assert_called_once_with(self.pte_path)

    def test_methods_raise_before_push_artifacts(self):
        """Every lifecycle method raises RuntimeError before push_artifacts."""
        request = GenerateRequest(prompts=["p"])
        for call in (
            self.adapter.get_metadata,
            lambda: self.adapter.push_inputs(request),
            lambda: self.adapter.execute(request),
            lambda: self.adapter.pull_results(request, self.tmp_dir),
        ):
            with self.assertRaisesRegex(RuntimeError, "push_artifacts"):
                call()

    def test_push_inputs_needs_nothing_for_generate(self):
        """A generation request's prompts travel as flags, so nothing is pushed."""
        self._push_artifacts()
        self.adapter.push_inputs(GenerateRequest(prompts=["p"]))
        self.platform.add_file.assert_not_called()

    def test_push_inputs_writes_forward_tokens(self):
        """A forward request's tokens are written as uint64 and added for the runner."""
        self._push_artifacts()
        self.adapter.push_inputs(ForwardRequest(token_ids=[3, 1, 4]))
        token_path = self.artifact_dir / RUNNER_INPUT_TOKENS
        self.platform.add_file.assert_called_once_with(token_path)
        np.testing.assert_array_equal(
            np.fromfile(token_path, dtype=np.uint64), [3, 1, 4]
        )

    def test_push_inputs_rejects_tokens_beyond_context(self):
        """Tokens that do not fit the PTE context raise ValueError."""
        self._push_artifacts()
        with self.assertRaisesRegex(ValueError, "max context length"):
            self.adapter.push_inputs(ForwardRequest(token_ids=list(range(8))))

    def test_push_inputs_rejects_unsupported_request(self):
        """An ExecuteRequest is not something the llama runner serves."""
        self._push_artifacts()
        with self.assertRaises(TypeError):
            self.adapter.push_inputs(ExecuteRequest(inputs=((1,),)))

    def test_execute_generate_builds_generation_flags(self):
        """A generation request runs the decoder with the generation flags."""
        self._push_artifacts()
        self.adapter.execute(GenerateRequest(prompts=["hi there"]))
        args = self._executed_args()
        self.assertEqual(
            args[:5],
            [
                "--decoder_model_version llama2",
                "--tokenizer_path tokenizer.bin",
                f"--output_path /out/{RUNNER_OUTPUTS}",
                f"--performance_output_path /out/{RUNNER_INFERENCE_SPEED}",
                f"--model_path {self.pte_path.name}",
            ],
        )
        self.assertIn("--eval_mode 0", args)
        self.assertIn("--seq_len 64", args)
        self.assertIn("--prompt 'hi there'", args)

    def test_execute_generate_caps_seq_len_on_limited_platform(self):
        """A platform that limits generation caps the text runner at 32 tokens."""
        self.platform.limits_generation = True
        self._push_artifacts()
        self.adapter.execute(GenerateRequest(prompts=["p"]))
        self.assertIn("--seq_len 32", self._executed_args())

    def test_execute_forward_runs_in_kv_mode(self):
        """A forward request always runs in KV mode and dumps the logits."""
        self.context = make_test_context(
            model_name="stories110m", extra_options={"model_mode": "hybrid"}
        )
        self._push_artifacts()
        self.adapter.execute(ForwardRequest(token_ids=[3, 1, 4]))
        args = self._executed_args()
        self.assertEqual(
            args[-5:],
            [
                "--eval_mode 0",
                "--temperature 0",
                "--seq_len 4",
                f"--dump_logits_path /out/{RUNNER_LOGITS}",
                f"--tokenized_prompt {RUNNER_INPUT_TOKENS}",
            ],
        )

    def test_execute_forward_honours_seq_len(self):
        """An explicit decode bound replaces the default token count plus one."""
        self._push_artifacts()
        self.adapter.execute(ForwardRequest(token_ids=[3, 1], seq_len=6))
        self.assertIn("--seq_len 6", self._executed_args())

    def test_execute_rejects_unsupported_request(self):
        """An ExecuteRequest raises TypeError."""
        self._push_artifacts()
        with self.assertRaises(TypeError):
            self.adapter.execute(ExecuteRequest(inputs=((1,),)))

    def test_pull_results_reads_generated_text(self):
        """A generation request's text and speed are pulled and decoded."""
        (self.runner_dir / RUNNER_OUTPUTS).write_text("generated")
        (self.runner_dir / RUNNER_INFERENCE_SPEED).write_text("7.5")
        self._push_artifacts()
        output_dir = self.tmp_dir / "collected"
        result = self.adapter.pull_results(GenerateRequest(prompts=["p"]), output_dir)
        self.assertIsInstance(result, GenerateOutput)
        self.assertEqual(result.texts, ["generated"])
        self.assertEqual(result.tokens_per_sec, 7.5)
        self.assertEqual(
            result.files,
            [output_dir / RUNNER_OUTPUTS, output_dir / RUNNER_INFERENCE_SPEED],
        )

    def test_pull_results_dequantizes_forward_logits(self):
        """A forward request's logits are dequantized and cut to its tokens."""
        np.arange(12, dtype=np.uint16).tofile(self.runner_dir / RUNNER_LOGITS)
        (self.runner_dir / RUNNER_INFERENCE_SPEED).write_text("3.0")
        self._push_artifacts()
        output_dir = self.tmp_dir / "collected"
        result = self.adapter.pull_results(ForwardRequest(token_ids=[1, 2]), output_dir)
        self.assertIsInstance(result, ForwardOutput)
        expected = (torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) - 2) * 0.5
        torch.testing.assert_close(result.logits, expected)
        self.assertEqual(result.tokens_per_sec, 3.0)
        self.assertEqual(
            set(result.files),
            {output_dir / RUNNER_LOGITS, output_dir / RUNNER_INFERENCE_SPEED},
        )


if __name__ == "__main__":
    unittest.main()
