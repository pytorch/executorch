# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline import cli
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
    InferenceResult,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
    DefaultCompilerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.llm_runner_adapter import (
    LLMRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.mllm_runner_adapter import (
    MLLMRunnerAdapter,
)

_CLI = "executorch.backends.qualcomm.genai_pipeline.cli"
_MODEL_LOOKUP = "executorch.backends.qualcomm.genai_pipeline.model_lookup"
_LOGGER = "executorch.backends.qualcomm.genai_pipeline.cli"


def _make_args(*extra, model="stories260k"):
    """Parse CLI arguments for ``model`` on SM8650 plus ``extra`` flags."""
    return cli.build_parser().parse_args(["--model", model, "--soc", "SM8650", *extra])


def _make_eval_args(model_mode, eval_method, *extra):
    """Parse CLI arguments requesting ``eval_method`` in ``model_mode``."""
    return _make_args(
        "--model-mode",
        model_mode,
        "--eval-methods",
        eval_method,
        "--eval-tasks",
        "wikitext",
        *extra,
    )


_VLM_CONFIG = MagicMock(spec=["vision_encoder"], name="vlm_config")


def _patch_model_lookups(model_config=None):
    """Patch every model lookup ``_prepare_configs`` performs."""
    return patch.multiple(
        _MODEL_LOOKUP,
        get_model_config=MagicMock(
            return_value=model_config or MagicMock(name="model_config")
        ),
        get_model_arch=MagicMock(return_value=MagicMock(name="model_arch")),
        get_source_transform=MagicMock(return_value=([], [])),
        get_model_num_sharding=MagicMock(return_value=1),
        get_state_dict_loader=MagicMock(return_value=MagicMock(name="loader")),
        get_quant_dtype=MagicMock(return_value=None),
        get_quant_recipe=MagicMock(return_value={}),
    )


class TestBuildParser(unittest.TestCase):
    def test_parses_model_soc_and_pre_gen_pte(self):
        """The parser keeps the model selector, SoC and pre-generated PTE directory."""
        args = _make_args("--pre-gen-pte", "/tmp/ptes")

        self.assertEqual(args.model, "stories260k")
        self.assertEqual(args.soc, "SM8650")
        self.assertEqual(args.pre_gen_pte, "/tmp/ptes")


class TestValidateEvalArgs(unittest.TestCase):
    def test_drops_repeated_eval_methods(self):
        """Repeated methods are dropped in place, keeping the first occurrence."""
        args = _make_args("--eval-methods", "prompt_eval", "prompt_eval")

        with self.assertLogs(_LOGGER, level="WARNING"):
            cli._validate_eval_args(args)

        self.assertEqual(args.eval_methods, ["prompt_eval"])

    def test_tasks_eval_rejects_lookahead_mode(self):
        """tasks_eval accepts only kv and hybrid, as the legacy flow does."""
        args = _make_eval_args("lookahead", "tasks_eval")

        with self.assertRaisesRegex(ValueError, "only supported for KV mode"):
            cli._validate_eval_args(args)

    def test_non_kv_modes_warn_that_only_the_forwards_run_in_kv(self):
        """hybrid tasks_eval and non-kv sqnr_eval warn that their forwards run in kv mode."""
        for model_mode, eval_method in (
            ("hybrid", "tasks_eval"),
            ("hybrid", "sqnr_eval"),
            ("lookahead", "sqnr_eval"),
        ):
            with self.subTest(model_mode=model_mode, eval_method=eval_method):
                args = _make_eval_args(model_mode, eval_method)

                with self.assertLogs(_LOGGER, level="WARNING") as logs:
                    cli._validate_eval_args(args)

                self.assertIn(
                    f"kv mode; other methods keep {model_mode}", logs.output[0]
                )
                self.assertEqual(args.model_mode, model_mode)

    def test_kv_mode_does_not_warn(self):
        """kv mode needs no mode switch for either evaluation method."""
        for eval_method in ("tasks_eval", "sqnr_eval"):
            with self.subTest(eval_method=eval_method):
                args = _make_eval_args("kv", eval_method)

                with patch(f"{_CLI}.logger") as logger:
                    cli._validate_eval_args(args)

                logger.warning.assert_not_called()

    def test_tasks_eval_requires_eval_tasks(self):
        """tasks_eval cannot run without --eval-tasks."""
        args = _make_args("--eval-methods", "tasks_eval")

        with self.assertRaisesRegex(ValueError, "--eval-tasks is required"):
            cli._validate_eval_args(args)

    def test_sqnr_eval_takes_one_prompt(self):
        """sqnr_eval compares one prompt's logits."""
        args = _make_eval_args("kv", "sqnr_eval", "--prompt", "a", "b")

        with self.assertRaisesRegex(ValueError, "exactly one --prompt"):
            cli._validate_eval_args(args)

    def test_sqnr_eval_takes_batch_size_one(self):
        """sqnr_eval supports only a batch size of one."""
        args = _make_eval_args("kv", "sqnr_eval", "--batch-size", "2")

        with self.assertRaisesRegex(ValueError, "--batch-size 1"):
            cli._validate_eval_args(args)


class TestValidateMultimodalArgs(unittest.TestCase):
    def test_text_model_rejects_media(self):
        """Media inputs require a multimodal model."""
        args = _make_args("--image-path", "cat.png")

        with self.assertRaisesRegex(ValueError, "require a multimodal model"):
            cli._validate_multimodal_args(args)

    def test_multimodal_model_forces_batch_size_one(self):
        """A multimodal model's batch size is forced to one in place."""
        args = _make_args(
            "--batch-size", "4", "--compile-only", model="smolvlm_500m_instruct"
        )

        with self.assertLogs(_LOGGER, level="WARNING"):
            cli._validate_multimodal_args(args)

        self.assertEqual(args.batch_size, 1)

    def test_compile_only_skips_media_validation(self):
        """Media inputs are checked only when inference runs."""
        args = _make_args("--compile-only", model="smolvlm_500m_instruct")

        with patch(f"{_MODEL_LOOKUP}.validate_modality_inputs") as validate:
            cli._validate_multimodal_args(args)

        validate.assert_not_called()

    def test_inference_validates_media_for_the_model(self):
        """With inference, the media inputs are checked against the model."""
        args = _make_args("--image-path", "cat.png", model="smolvlm_500m_instruct")

        with patch(f"{_MODEL_LOOKUP}.validate_modality_inputs") as validate:
            cli._validate_multimodal_args(args)

        validate.assert_called_once_with("smolvlm_500m_instruct", ["cat.png"], [])


class TestSkipsModelPreparation(unittest.TestCase):
    def test_skipped_only_with_pre_gen_pte_and_no_logit_method(self):
        """tasks_eval and sqnr_eval keep model preparation for pre-generated PTEs."""
        for extra, expected in (
            ((), False),
            (("--pre-gen-pte", "/tmp/ptes"), True),
            (("--pre-gen-pte", "/tmp/ptes", "--eval-methods", "tasks_eval"), False),
            (("--pre-gen-pte", "/tmp/ptes", "--eval-methods", "sqnr_eval"), False),
        ):
            with self.subTest(extra=extra):
                self.assertIs(
                    cli._skips_model_preparation(_make_args(*extra)), expected
                )


class TestPrepareConfigs(unittest.TestCase):
    def test_runtime_tokenizer_path_is_top_level_not_an_inference_option(self):
        """The tokenizer override travels apart from the runner's inference options."""
        args = _make_args("--runtime-tokenizer-path", "tok.bin")

        with _patch_model_lookups():
            configs = cli._prepare_configs(args)

        self.assertEqual(configs["runtime_tokenizer_path"], "tok.bin")
        self.assertNotIn("runtime_tokenizer_path", configs["inference_options"])

    def test_explicit_runtime_tokenizer_path_wins(self):
        """An explicit path is used even when model preparation is skipped."""
        args = _make_args(
            "--runtime-tokenizer-path", "tok.bin", "--pre-gen-pte", "/tmp/ptes"
        )

        with _patch_model_lookups(), patch(
            f"{_CLI}._load_runtime_tokenizer_path"
        ) as load:
            configs = cli._prepare_configs(args)

        load.assert_not_called()
        self.assertEqual(configs["runtime_tokenizer_path"], "tok.bin")

    def test_resolves_runtime_tokenizer_when_model_preparation_is_skipped(self):
        """Without model preparation, the CLI resolves the tokenizer itself."""
        args = _make_args("--pre-gen-pte", "/tmp/ptes")

        with _patch_model_lookups(), patch(
            f"{_CLI}._load_runtime_tokenizer_path", return_value="resolved.json"
        ) as load:
            configs = cli._prepare_configs(args)

        load.assert_called_once()
        self.assertEqual(configs["runtime_tokenizer_path"], "resolved.json")

    def test_multi_image_skips_encoder_quantization(self):
        """More than one image keeps the vision encoder unquantized, as legacy does."""
        args = _make_args(
            "--image-path", "a.png", "b.png", model="smolvlm_500m_instruct"
        )

        with _patch_model_lookups(_VLM_CONFIG):
            configs = cli._prepare_configs(args)

        self.assertEqual(
            configs["quantize_options"]["skip_quantize"], ["vision_encoder"]
        )

    def test_single_image_quantizes_the_encoder(self):
        """One image leaves every component quantized."""
        args = _make_args("--image-path", "a.png", model="smolvlm_500m_instruct")

        with _patch_model_lookups(_VLM_CONFIG):
            configs = cli._prepare_configs(args)

        self.assertNotIn("skip_quantize", configs["quantize_options"])

    def test_leaves_runtime_tokenizer_to_model_preparation(self):
        """With model preparation, no override is set and nothing is resolved."""
        args = _make_args("--use-fp16")

        with _patch_model_lookups(), patch(
            f"{_CLI}._load_runtime_tokenizer_path"
        ) as load:
            configs = cli._prepare_configs(args)

        load.assert_not_called()
        self.assertIsNone(configs["runtime_tokenizer_path"])


class TestStageFactories(unittest.TestCase):
    def test_pre_gen_pte_skips_quantization_and_compilation(self):
        """Pre-generated PTEs need neither quantization nor compilation."""
        args = _make_args("--pre-gen-pte", "/tmp/ptes")

        self.assertIsNone(cli.get_quantization_stage(args, {}, {}))
        self.assertIsNone(cli.get_compilation_stage(args))

    def test_pre_gen_pte_skips_model_preparation(self):
        """Model preparation is skipped when no evaluation method needs it."""
        args = _make_args("--pre-gen-pte", "/tmp/ptes")

        self.assertIsNone(cli.get_model_preparation_stage(args))

    def test_fp16_skips_quantization(self):
        """FP16 inference skips quantization."""
        self.assertIsNone(cli.get_quantization_stage(_make_args("--use-fp16"), {}, {}))

    def test_compile_only_skips_inference(self):
        """Compile-only builds no inference stage."""
        self.assertIsNone(cli.get_inference_stage(_make_args("--compile-only")))

    def test_compilation_stage_uses_default_compiler_adapter(self):
        """The compilation stage compiles through the default compiler adapter."""
        stage = cli.get_compilation_stage(_make_args())

        self.assertIsInstance(stage._strategy.adapter, DefaultCompilerAdapter)

    def test_inference_stage_selects_runner_adapter_by_model_kind(self):
        """Text models get the LLM runner adapter, multimodal models the MLLM one."""
        for model, adapter_type in (
            ("stories260k", LLMRunnerAdapter),
            ("smolvlm_500m_instruct", MLLMRunnerAdapter),
            ("granite_speech_3_3-2b", MLLMRunnerAdapter),
        ):
            with self.subTest(model=model), patch(
                "executorch.backends.qualcomm.export_utils.QnnConfig.load_config"
            ), patch(
                "executorch.backends.qualcomm.genai_pipeline.inference.platforms.get_runner_platform"
            ):
                stage = cli.get_inference_stage(_make_args(model=model))

            self.assertIsInstance(stage._strategy.adapter, adapter_type)
            self.assertEqual(
                [evaluator.method for evaluator in stage._strategy.evaluators],
                ["prompt_eval"],
            )


class TestRunPipeline(unittest.TestCase):
    @patch(f"{_CLI}.get_inference_stage")
    @patch(f"{_CLI}.get_compilation_stage")
    @patch(f"{_CLI}.get_quantization_stage")
    @patch(f"{_CLI}.get_model_preparation_stage")
    @patch(f"{_CLI}._prepare_configs")
    @patch("executorch.backends.qualcomm.genai_pipeline.genai_pipeline.GenAIPipeline")
    def test_context_carries_runtime_tokenizer_path_beside_pre_gen_pte(
        self,
        pipeline_type,
        prepare_configs,
        get_model_preparation_stage,
        get_quantization_stage,
        get_compilation_stage,
        get_inference_stage,
    ):
        """The tokenizer override reaches the context's top-level extra_options."""
        result = InferenceOutputConfig()
        pipeline_type.return_value.invoke.return_value = result
        inference_options = {"temperature": 0.0}
        prepare_configs.return_value = {
            "dataset_options": MagicMock(name="dataset_options"),
            "model_options": MagicMock(name="model_options"),
            "quantize_options": MagicMock(name="quantize_options"),
            "inference_options": inference_options,
            "runtime_tokenizer_path": "tok.bin",
        }

        returned = cli.run_pipeline(_make_args("--pre-gen-pte", "/tmp/ptes"))

        self.assertIs(returned, result)
        context = pipeline_type.return_value.invoke.call_args.args[0]
        self.assertEqual(context.extra_options["pre_gen_pte"], "/tmp/ptes")
        self.assertEqual(context.extra_options["runtime_tokenizer_path"], "tok.bin")
        self.assertIs(context.extra_options["inference_options"], inference_options)
        self.assertTrue(context.extra_options["export_tokenizer"])
        kwargs = pipeline_type.call_args.kwargs
        self.assertIs(kwargs["compilation_stage"], get_compilation_stage.return_value)
        self.assertIs(kwargs["inference_stage"], get_inference_stage.return_value)


class TestShowInferenceResults(unittest.TestCase):
    def test_logs_generations_metrics_and_performance_per_method(self):
        """Each method's generations, evaluation results and throughput are logged."""
        result = InferenceOutputConfig(
            results={
                "prompt_eval": InferenceResult(
                    generations=["Once upon a time\n"],
                    performance_metrics={"num_requests": 1},
                ),
                "tasks_eval": InferenceResult(eval_results={"wiki_ppl": 12.5}),
            }
        )

        with self.assertLogs(_LOGGER, level="INFO") as logs:
            cli.show_inference_results(result)

        output = "\n".join(logs.output)
        self.assertIn("prompt_eval", output)
        self.assertIn("Output: Once upon a time", output)
        self.assertIn("num_requests", output)
        self.assertIn("tasks_eval", output)
        self.assertIn("wiki_ppl", output)

    def test_logs_nothing_without_results(self):
        """An empty output logs nothing."""
        with patch(f"{_CLI}.logger") as logger:
            cli.show_inference_results(InferenceOutputConfig())

        logger.info.assert_not_called()


class TestMain(unittest.TestCase):
    @patch(f"{_CLI}.show_inference_results")
    @patch(f"{_CLI}.run_pipeline")
    def test_shows_the_pipeline_results(self, run_pipeline, show_inference_results):
        """main runs the pipeline and shows what it returns."""
        cli.main(["--model", "Stories260K", "--soc", "SM8650"])

        self.assertEqual(run_pipeline.call_args.args[0].model, "stories260k")
        show_inference_results.assert_called_once_with(run_pipeline.return_value)

    def test_exits_with_code_one_on_failure(self):
        """A failing pipeline exits with status 1."""
        for error in (KeyError("unknown_model"), RuntimeError("device lost")):
            with self.subTest(error=type(error).__name__), patch(
                f"{_CLI}.run_pipeline", side_effect=error
            ), patch(f"{_CLI}.logger"), self.assertRaises(SystemExit) as raised:
                cli.main(["--model", "stories260k", "--soc", "SM8650"])

            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
