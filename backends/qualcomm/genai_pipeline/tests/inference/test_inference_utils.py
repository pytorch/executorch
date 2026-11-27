# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, create_autospec, patch

import numpy as np
import torch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
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
    RunnerMetadata,
)
from executorch.backends.qualcomm.genai_pipeline.inference.utils import (
    build_generate_args,
    build_runner_args,
    get_pte_paths,
    get_runtime_tokenizer_path,
    post_process_logits,
    preprocess_encoder_inputs,
    read_generate_output,
    retrieve_info_from_pte,
    RUNNER_INFERENCE_SPEED,
    RUNNER_OUTPUTS,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_SOC_CHIPSET,
)
from executorch.exir import to_edge
from executorch.exir._serialize._program import _get_extended_header


def _make_input_config(artifact_dir, **overrides):
    """Create an InferenceInputConfig rooted at ``artifact_dir``."""
    defaults = {"soc_model": TEST_SOC_CHIPSET, "artifact_dir": Path(artifact_dir)}
    defaults.update(overrides)
    return InferenceInputConfig(**defaults)


def _make_metadata(**overrides):
    """Create RunnerMetadata for a 4-token vocabulary."""
    defaults = {
        "max_context_len": 8,
        "max_seq_len": 8,
        "vocab_size": 4,
        "logits_scale": 0.5,
        "logits_zero_point": 2,
        "kv_io_bit_width": 16,
    }
    defaults.update(overrides)
    return RunnerMetadata(**defaults)


def _make_platform(**overrides):
    """Create a mock RunnerPlatform naming files by their bare name."""
    platform = create_autospec(RunnerPlatform, instance=True)
    platform.output_dir.return_value = "/out"
    platform.runner_path.side_effect = lambda path: Path(path).name
    platform.target_flags = ("--shared_buffer",)
    platform.limits_generation = False
    for name, value in overrides.items():
        setattr(platform, name, value)
    return platform


class TestGetPtePaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.decoder = self.tmp_dir / "m_text_decoder.pte"
        self.decoder.write_bytes(b"pte")

    def tearDown(self):
        self._tmp.cleanup()

    def test_selects_text_decoder_by_default(self):
        """Only the text decoder is selected when no keys are given."""
        pte_paths = {
            ARTIFACT_TEXT_DECODER: self.decoder,
            ARTIFACT_TOK_EMBEDDING: self.tmp_dir / "unused.pte",
        }
        self.assertEqual(
            get_pte_paths(pte_paths), {ARTIFACT_TEXT_DECODER: self.decoder}
        )

    def test_returns_path_objects(self):
        """String paths are returned as Path objects."""
        result = get_pte_paths({ARTIFACT_TEXT_DECODER: str(self.decoder)})
        self.assertEqual(result[ARTIFACT_TEXT_DECODER], self.decoder)

    def test_missing_key_raises(self):
        """A requested key absent from the mapping raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Missing PTE artifact"):
            get_pte_paths(
                {ARTIFACT_TEXT_DECODER: self.decoder},
                (ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING),
            )

    def test_not_a_file_raises(self):
        """A requested path that does not exist raises ValueError."""
        with self.assertRaisesRegex(ValueError, "is not a file"):
            get_pte_paths({ARTIFACT_TEXT_DECODER: self.tmp_dir / "absent.pte"})


class TestGetRuntimeTokenizerPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_none_raises(self):
        """A missing runtime tokenizer raises ValueError naming the CLI flag."""
        with self.assertRaisesRegex(ValueError, "--runtime-tokenizer-path"):
            get_runtime_tokenizer_path(_make_input_config(self.tmp_dir))

    def test_not_a_file_raises(self):
        """A runtime tokenizer that does not exist raises ValueError."""
        config = _make_input_config(
            self.tmp_dir, runtime_tokenizer_path=self.tmp_dir / "absent.json"
        )
        with self.assertRaisesRegex(ValueError, "not an existing file"):
            get_runtime_tokenizer_path(config)

    def test_returns_existing_file(self):
        """An existing runtime tokenizer is returned as a Path."""
        tokenizer = self.tmp_dir / "tokenizer.json"
        tokenizer.write_text("{}")
        config = _make_input_config(self.tmp_dir, runtime_tokenizer_path=str(tokenizer))
        self.assertEqual(get_runtime_tokenizer_path(config), tokenizer)


def _write_pte(path, constants, weight_numel=2):
    """Write a PTE whose constant methods return ``constants``."""

    class AddWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(weight_numel))

        def forward(self, x):
            return x + self.weight

    program = to_edge(
        torch.export.export(AddWeight(), (torch.ones(weight_numel),)),
        constant_methods=constants,
    ).to_executorch()
    path.write_bytes(program.buffer)
    return path


class TestRetrieveInfoFromPte(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pte_path = Path(self._tmp.name) / "m.pte"

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_a_quantized_pte(self):
        """The constant methods map onto the RunnerMetadata fields."""
        _write_pte(
            self.pte_path,
            {
                "get_vocab_size": 32000,
                "get_max_seq_len": 64,
                "get_max_context_len": 128,
                "get_logits_scale": 0.1,
                "get_logits_zero_point": 3,
                "get_kv_io_bit_width": 8,
            },
        )
        self.assertEqual(
            retrieve_info_from_pte(self.pte_path),
            RunnerMetadata(
                max_context_len=128,
                max_seq_len=64,
                vocab_size=32000,
                logits_scale=0.1,
                logits_zero_point=3,
                kv_io_bit_width=8,
            ),
        )

    def test_float_pte_defaults(self):
        """A float PTE has 32-bit IO, identity logits, and context of max_seq_len."""
        _write_pte(self.pte_path, {"get_vocab_size": 4, "get_max_seq_len": 8})
        self.assertEqual(
            retrieve_info_from_pte(self.pte_path),
            RunnerMetadata(
                max_context_len=8,
                max_seq_len=8,
                vocab_size=4,
                logits_scale=1.0,
                logits_zero_point=0,
                kv_io_bit_width=32,
            ),
        )

    def test_missing_vocab_size_raises(self):
        """A PTE without the decoder's vocab size is rejected."""
        _write_pte(self.pte_path, {"get_max_seq_len": 8})
        with self.assertRaisesRegex(RuntimeError, "vocab size"):
            retrieve_info_from_pte(self.pte_path)

    def test_reads_only_the_program_of_a_pte_with_segments(self):
        """The metadata is read from the flatbuffer, ahead of the data segments."""
        _write_pte(
            self.pte_path,
            {"get_vocab_size": 4, "get_max_seq_len": 8},
            weight_numel=1 << 16,
        )
        header = _get_extended_header(self.pte_path.read_bytes())
        self.assertLess(header.program_size, self.pte_path.stat().st_size)
        self.assertGreater(header.segment_base_offset, 0)
        self.assertEqual(retrieve_info_from_pte(self.pte_path).vocab_size, 4)

    def test_quantized_pte_without_logits_scale_raises(self):
        """A quantized PTE must carry the scale its logits are dequantized with."""
        _write_pte(
            self.pte_path,
            {"get_vocab_size": 4, "get_max_seq_len": 8, "get_kv_io_bit_width": 16},
        )
        with self.assertRaisesRegex(RuntimeError, "scale/offset"):
            retrieve_info_from_pte(self.pte_path)


class TestPostProcessLogits(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.logits_path = Path(self._tmp.name) / "all_logits.raw"

    def tearDown(self):
        self._tmp.cleanup()

    def test_dequantizes_uint16_logits(self):
        """Quantized logits are dequantized with the PTE's scale and zero point."""
        np.arange(12, dtype=np.uint16).tofile(self.logits_path)
        logits = post_process_logits(self.logits_path, _make_metadata(), num_tokens=3)
        expected = (torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) - 2) * 0.5
        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits, expected)

    def test_reads_float32_logits_as_is(self):
        """Float logits with unit scale and zero zero-point are returned unchanged."""
        values = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        values.tofile(self.logits_path)
        metadata = _make_metadata(
            logits_scale=1.0, logits_zero_point=0, kv_io_bit_width=32
        )
        logits = post_process_logits(self.logits_path, metadata, num_tokens=2)
        torch.testing.assert_close(logits, torch.from_numpy(values).reshape(1, 2, 4))

    def test_keeps_only_the_prompt_positions(self):
        """Positions the runner decoded past the prompt are cut off."""
        np.arange(12, dtype=np.uint16).tofile(self.logits_path)
        logits = post_process_logits(self.logits_path, _make_metadata(), num_tokens=1)
        self.assertEqual(tuple(logits.shape), (1, 1, 4))


_LEGACY_DATASET = "executorch.examples.qualcomm.oss_scripts.llama.dataset"


class TestPreprocessEncoderInputs(unittest.TestCase):
    def setUp(self):
        from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
            AudioModalityConfig,
            VisionModalityConfig,
        )

        self.vision_config = type("Vision", (VisionModalityConfig,), {})
        self.audio_config = type("Audio", (AudioModalityConfig,), {})
        patchers = {
            "preprocessor_cls": patch(
                f"{_LEGACY_DATASET}.preprocessors.ModalityPreprocessor"
            ),
            "load_image": patch(
                "transformers.image_utils.load_image", side_effect=lambda p: f"img:{p}"
            ),
            "load_audio": patch(
                f"{_LEGACY_DATASET}.loaders.load_audio_file",
                side_effect=lambda p, repo_id: f"wav:{p}",
            ),
        }
        for name, patcher in patchers.items():
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self.preprocessor = self.preprocessor_cls.return_value

    def test_preprocesses_every_image_with_one_token_each(self):
        """All images go through one vision call, prompted with a token per image."""
        config = SimpleNamespace(repo_id="repo", vision_encoder=self.vision_config)

        result = preprocess_encoder_inputs(
            config, "smolvlm_500m_instruct", [], ["a.png", "b.png"]
        )

        self.preprocessor_cls.assert_called_once_with("repo")
        self.preprocessor.preprocess_images.assert_called_once_with(
            self.vision_config, "<image><image>", ["img:a.png", "img:b.png"]
        )
        self.assertEqual(
            result,
            {ARTIFACT_VISION_ENCODER: self.preprocessor.preprocess_images.return_value},
        )

    def test_preprocesses_each_audio_file_separately(self):
        """Each audio file gets its own call, prompted with a token per file."""
        config = SimpleNamespace(repo_id="repo", audio_encoder=self.audio_config)

        result = preprocess_encoder_inputs(
            config, "granite_speech_3_3-2b", ["a.wav", "b.wav"], []
        )

        self.assertEqual(
            self.preprocessor.preprocess_audio.call_args_list,
            [
                call(self.audio_config, "<|audio|><|audio|>", "wav:a.wav"),
                call(self.audio_config, "<|audio|><|audio|>", "wav:b.wav"),
            ],
        )
        self.assertEqual(len(result[ARTIFACT_AUDIO_ENCODER]), 2)

    def test_skips_a_modality_without_paths_or_config(self):
        """Paths for a modality the model lacks, or no paths at all, add nothing."""
        config = SimpleNamespace(repo_id="repo", vision_encoder=self.vision_config)

        result = preprocess_encoder_inputs(
            config, "smolvlm_500m_instruct", ["a.wav"], []
        )

        self.assertEqual(result, {})
        self.preprocessor.preprocess_images.assert_not_called()
        self.preprocessor.preprocess_audio.assert_not_called()


class TestBuildRunnerArgs(unittest.TestCase):
    def test_builds_shared_flags(self):
        """The shared flags name the model version, tokenizer, outputs, and target."""
        control_args = ControlArgs(decoder_model="stories110m")
        args = build_runner_args(
            control_args, _make_platform(), Path("/host/tokenizer.bin")
        )
        self.assertEqual(
            args,
            [
                "--decoder_model_version llama2",
                "--tokenizer_path tokenizer.bin",
                f"--output_path /out/{RUNNER_OUTPUTS}",
                f"--performance_output_path /out/{RUNNER_INFERENCE_SPEED}",
                "--shared_buffer",
            ],
        )

    def test_quotes_paths_with_spaces(self):
        """Host paths survive the shell the platform runs the command through."""
        platform = _make_platform()
        platform.output_dir.return_value = "/data/my runs"
        platform.runner_path.side_effect = str
        args = build_runner_args(
            ControlArgs(decoder_model="stories110m"),
            platform,
            Path("/my dir/tokenizer.bin"),
        )
        self.assertIn("--tokenizer_path '/my dir/tokenizer.bin'", args)
        self.assertIn(f"--output_path '/data/my runs/{RUNNER_OUTPUTS}'", args)


class TestBuildGenerateArgs(unittest.TestCase):
    def test_kv_mode_quotes_prompts(self):
        """User-supplied text is shell-quoted, one --prompt per prompt."""
        control_args = ControlArgs(
            model_mode="kv", max_seq_len=128, temperature=0.0, system_prompt="be nice"
        )
        args = build_generate_args(
            control_args, ["hi there", "bye"], _make_platform(), 32
        )
        self.assertEqual(
            args,
            [
                "--eval_mode 0",
                "--temperature 0.0",
                "--system_prompt 'be nice'",
                "--seq_len 128",
                "--prompt 'hi there'",
                "--prompt bye",
            ],
        )

    def test_limited_platform_caps_seq_len(self):
        """A platform that limits generation caps the decode length."""
        control_args = ControlArgs(model_mode="kv", max_seq_len=128)
        args = build_generate_args(
            control_args, ["p"], _make_platform(limits_generation=True), 32
        )
        self.assertIn("--seq_len 32", args)

    def test_lookahead_adds_lookahead_flags(self):
        """Lookahead mode adds the window, gcap, and ngram flags."""
        control_args = ControlArgs(model_mode="lookahead", window=4, gcap=5, ngram=3)
        args = build_generate_args(control_args, ["p"], _make_platform(), 32)
        self.assertIn("--eval_mode 2", args)
        self.assertEqual(args[-3:], ["--window 4", "--gcap 5", "--ngram 3"])


class TestReadGenerateOutput(unittest.TestCase):
    def test_reads_text_and_speed(self):
        """The runner text and speed files become a GenerateOutput."""
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / RUNNER_OUTPUTS
            speed = Path(tmp) / RUNNER_INFERENCE_SPEED
            outputs.write_text("hello")
            speed.write_text("12.5")
            result = read_generate_output(outputs, speed)
        self.assertEqual(result.texts, ["hello"])
        self.assertEqual(result.tokens_per_sec, 12.5)
        self.assertEqual(result.files, [outputs, speed])


if __name__ == "__main__":
    unittest.main()
