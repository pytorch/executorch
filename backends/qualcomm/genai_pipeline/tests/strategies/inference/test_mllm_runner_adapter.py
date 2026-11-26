# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import create_autospec, patch

import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
    RunnerPlatform,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ForwardRequest,
    GenerateRequest,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    RUNNER_INFERENCE_SPEED,
    RUNNER_OUTPUTS,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.mllm_runner_adapter import (
    MLLMRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)

_MODEL = "smolvlm_500m_instruct"
_VALIDATE = (
    "executorch.backends.qualcomm.genai_pipeline.model_lookup.validate_modality_inputs"
)
_PREPROCESS = (
    "executorch.backends.qualcomm.genai_pipeline.strategies.inference."
    "mllm_runner_adapter.preprocess_encoder_inputs"
)


def _make_platform(runner_dir):
    """Create a mock RunnerPlatform whose runner outputs live in ``runner_dir``."""
    platform = create_autospec(RunnerPlatform, instance=True)
    platform.target_flags = ()
    platform.limits_generation = False
    platform.output_dir.return_value = "/out"
    platform.runner_path.side_effect = lambda path: Path(path).name
    platform.runner_input_list.side_effect = lambda path: f"runner/{path.name}"
    platform.pull.side_effect = lambda filename, host_path: shutil.copy(
        runner_dir / filename, host_path
    )
    return platform


class TestMLLMRunnerAdapter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.artifact_dir = self.tmp_dir / "artifacts"
        self.runner_dir = self.tmp_dir / "runner"
        self.runner_dir.mkdir()
        self.pte_paths = {}
        for key in (
            ARTIFACT_TEXT_DECODER,
            ARTIFACT_TOK_EMBEDDING,
            ARTIFACT_VISION_ENCODER,
        ):
            self.pte_paths[key] = self.tmp_dir / f"{_MODEL}_{key}.pte"
            self.pte_paths[key].write_bytes(b"pte")
        self.tokenizer_path = self.tmp_dir / "tokenizer.json"
        self.tokenizer_path.write_text("{}")
        self.platform = _make_platform(self.runner_dir)
        self.adapter = MLLMRunnerAdapter(self.platform)
        self.control_args = ControlArgs(
            decoder_model=_MODEL,
            model_mode="kv",
            max_seq_len=512,
            image_path=["/host/cat.png"],
        )

        validate_patcher = patch(_VALIDATE)
        self.mock_validate = validate_patcher.start()
        self.addCleanup(validate_patcher.stop)
        preprocess_patcher = patch(
            _PREPROCESS,
            return_value={
                ARTIFACT_VISION_ENCODER: [(torch.ones(2, dtype=torch.float32),)]
            },
        )
        self.mock_preprocess = preprocess_patcher.start()
        self.addCleanup(preprocess_patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _input_config(self, **overrides):
        """Create an InferenceInputConfig holding the MLLM artifacts and one image."""
        defaults = {
            "soc_model": TEST_SOC_CHIPSET,
            "artifact_dir": self.artifact_dir,
            "pte_paths": self.pte_paths,
            "runtime_tokenizer_path": self.tokenizer_path,
        }
        defaults.update(overrides)
        return InferenceInputConfig(**defaults)

    def _push_artifacts(self, **overrides):
        """Deploy the adapter with the default input configuration."""
        self.adapter.push_artifacts(self.control_args, self._input_config(**overrides))

    def test_push_artifacts_validates_and_preprocesses_media(self):
        """The image paths are validated and preprocessed for the vision encoder."""
        self._push_artifacts()
        self.mock_validate.assert_called_once_with(_MODEL, ["/host/cat.png"], [])
        self.assertEqual(
            self.mock_preprocess.call_args.args[1:], (_MODEL, [], ["/host/cat.png"])
        )

    def test_push_artifacts_sets_up_ptes_tokenizer_and_inputs(self):
        """The PTEs, tokenizer, input list, and input files are handed to the platform."""
        self._push_artifacts()
        runner, files, host_dir = self.platform.setup.call_args.args
        input_list = self.artifact_dir / f"{ARTIFACT_VISION_ENCODER}_input_list.txt"
        self.assertEqual(runner, MLLMRunnerAdapter.RUNNER)
        self.assertEqual(host_dir, self.artifact_dir / "outputs")
        self.assertEqual(files[:3], list(self.pte_paths.values()))
        self.assertEqual(files[3:5], [self.tokenizer_path, input_list])
        self.assertEqual(
            [file.name for file in files[5:]],
            [f"{ARTIFACT_VISION_ENCODER}_input_0_0.raw"],
        )
        self.platform.runner_input_list.assert_called_once_with(input_list)

    def test_push_artifacts_requires_every_component(self):
        """A missing modality encoder PTE raises ValueError."""
        pte_paths = dict(self.pte_paths)
        del pte_paths[ARTIFACT_VISION_ENCODER]
        with self.assertRaisesRegex(ValueError, "Missing PTE artifact"):
            self._push_artifacts(pte_paths=pte_paths)

    def test_push_artifacts_requires_modality_inputs(self):
        """Media that produce no encoder inputs raise ValueError."""
        self.mock_preprocess.return_value = {}
        with self.assertRaisesRegex(ValueError, "No inputs produced"):
            self._push_artifacts()

    def test_get_metadata_is_none(self):
        """The multimodal flow reports no decoder metadata."""
        self.assertIsNone(self.adapter.get_metadata())

    def test_rejects_non_generate_requests(self):
        """Only generation requests are served."""
        self._push_artifacts()
        request = ForwardRequest(token_ids=[1])
        for call in (
            lambda: self.adapter.push_inputs(request),
            lambda: self.adapter.execute(request),
        ):
            with self.assertRaises(TypeError):
                call()

    def test_methods_raise_before_push_artifacts(self):
        """execute and pull_results raise RuntimeError before push_artifacts."""
        request = GenerateRequest(prompts=["p"])
        for call in (
            lambda: self.adapter.execute(request),
            lambda: self.adapter.pull_results(request, self.tmp_dir),
        ):
            with self.assertRaisesRegex(RuntimeError, "push_artifacts"):
                call()

    def test_execute_passes_component_paths_and_input_list(self):
        """The runner gets each component's PTE and the modality input list."""
        self._push_artifacts()
        self.adapter.push_inputs(GenerateRequest(prompts=["describe"]))
        self.adapter.execute(GenerateRequest(prompts=["describe"]))
        args = self.platform.execute.call_args.args[0]
        names = {key: path.name for key, path in self.pte_paths.items()}
        self.assertIn(f"--decoder_path {names[ARTIFACT_TEXT_DECODER]}", args)
        self.assertIn(f"--encoder_path {names[ARTIFACT_VISION_ENCODER]}", args)
        self.assertIn(f"--tok_embedding_path {names[ARTIFACT_TOK_EMBEDDING]}", args)
        self.assertIn(
            f"--image_path runner/{ARTIFACT_VISION_ENCODER}_input_list.txt", args
        )
        self.assertIn("--prompt describe", args)
        self.assertIn("--seq_len 512", args)

    def test_execute_caps_seq_len_on_limited_platform(self):
        """A platform that limits generation caps the multimodal runner at 320 tokens."""
        self.platform.limits_generation = True
        self._push_artifacts()
        self.adapter.execute(GenerateRequest(prompts=["p"]))
        self.assertIn("--seq_len 320", self.platform.execute.call_args.args[0])

    def test_pull_results_reads_generated_text(self):
        """The generated text and speed are pulled and decoded."""
        (self.runner_dir / RUNNER_OUTPUTS).write_text("a cat")
        (self.runner_dir / RUNNER_INFERENCE_SPEED).write_text("4.0")
        self._push_artifacts()
        result = self.adapter.pull_results(
            GenerateRequest(prompts=["p"]), self.tmp_dir / "collected"
        )
        self.assertEqual(result.texts, ["a cat"])
        self.assertEqual(result.tokens_per_sec, 4.0)


if __name__ == "__main__":
    unittest.main()
