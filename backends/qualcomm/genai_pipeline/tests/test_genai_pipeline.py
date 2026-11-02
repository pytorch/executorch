# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
    InferenceResult,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
from executorch.backends.qualcomm.genai_pipeline.genai_pipeline import GenAIPipeline
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    EngineType,
    STAGE_COMPILATION,
    STAGE_INFERENCE,
    STAGE_MODEL_PREPARATION,
    STAGE_QUANTIZATION,
)
from executorch.backends.qualcomm.genai_pipeline.stages.compilation_stage import (
    CompilationStage,
)
from executorch.backends.qualcomm.genai_pipeline.stages.inference_stage import (
    InferenceStage,
)
from executorch.backends.qualcomm.genai_pipeline.stages.model_preparation_stage import (
    ModelPreparationStage,
)
from executorch.backends.qualcomm.genai_pipeline.stages.quantization_stage import (
    QuantizationStage,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_ARTIFACT_PATHS,
    TEST_BACKEND_TYPE,
)

TEST_MOCK_GENERATED_TEXT = "Mock generated text"
TEST_MOCK_TOKENS_PER_SEC = 42.0
TEST_EVAL_METHOD = "prompt_eval"


class _MockQuantizationStrategy(QuantizationStrategy):
    def invoke(self, context, input_config):
        return QuantizationOutputConfig(graphs={"mock": "graphs"})


class _MockCompilationStrategy(CompilationStrategy):
    def invoke(self, context, input_config):
        return CompilationOutputConfig(artifact_paths=TEST_ARTIFACT_PATHS)


class _MockInferenceStrategy(InferenceStrategy):
    def invoke(self, context, input_config):
        return InferenceOutputConfig(
            results={
                TEST_EVAL_METHOD: InferenceResult(
                    generations=[TEST_MOCK_GENERATED_TEXT],
                    performance_metrics={"tokens_per_sec": TEST_MOCK_TOKENS_PER_SEC},
                )
            }
        )


class TestGenAIPipelineFromProxy(unittest.TestCase):

    def test_full_executorch_creates_all_stages(self):
        proxy = EngineProxy(
            {
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
                STAGE_INFERENCE: EngineType.EXECUTORCH,
            },
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline.from_proxy(proxy)
        self.assertIsNotNone(pipeline._model_preparation_stage)
        self.assertIsNotNone(pipeline._quantization_stage)
        self.assertIsNotNone(pipeline._compilation_stage)
        self.assertIsNotNone(pipeline._inference_stage)

    def test_skip_stages(self):
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH},
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline.from_proxy(
            proxy,
            skip_stages={
                STAGE_MODEL_PREPARATION,
                STAGE_QUANTIZATION,
                STAGE_COMPILATION,
                STAGE_INFERENCE,
            },
        )
        self.assertIsNone(pipeline._model_preparation_stage)
        self.assertIsNone(pipeline._quantization_stage)
        self.assertIsNone(pipeline._compilation_stage)
        self.assertIsNone(pipeline._inference_stage)

    def test_default_engines_all_executorch(self):
        proxy = EngineProxy({}, backend_type=TEST_BACKEND_TYPE)
        pipeline = GenAIPipeline.from_proxy(proxy)
        self.assertIsNotNone(pipeline._model_preparation_stage)
        self.assertIsNotNone(pipeline._quantization_stage)
        self.assertIsNotNone(pipeline._compilation_stage)
        self.assertIsNotNone(pipeline._inference_stage)

    def test_from_proxy_unsupported_model_preparation_engine_raises(self):
        proxy = MagicMock(spec=EngineProxy)
        proxy.get_engine.return_value = MagicMock()
        with self.assertRaises(ValueError) as cm:
            GenAIPipeline.from_proxy(proxy)
        self.assertIn("No model_preparation strategy", str(cm.exception))

    def test_from_proxy_unsupported_quantization_engine_raises(self):
        proxy = MagicMock(spec=EngineProxy)
        proxy.get_engine.side_effect = lambda stage: (
            EngineType.EXECUTORCH if stage == STAGE_MODEL_PREPARATION else MagicMock()
        )
        with self.assertRaises(ValueError) as cm:
            GenAIPipeline.from_proxy(proxy)
        self.assertIn("No quantization strategy", str(cm.exception))

    def test_from_proxy_unsupported_compilation_engine_raises(self):
        proxy = MagicMock(spec=EngineProxy)
        proxy.get_engine.side_effect = lambda stage: (
            EngineType.EXECUTORCH
            if stage in (STAGE_MODEL_PREPARATION, STAGE_QUANTIZATION)
            else MagicMock()
        )
        with self.assertRaises(ValueError) as cm:
            GenAIPipeline.from_proxy(proxy)
        self.assertIn("No compilation strategy", str(cm.exception))

    def test_from_proxy_unsupported_inference_engine_raises(self):
        proxy = MagicMock(spec=EngineProxy)
        proxy.get_engine.side_effect = lambda stage: (
            EngineType.EXECUTORCH
            if stage in (STAGE_MODEL_PREPARATION, STAGE_QUANTIZATION, STAGE_COMPILATION)
            else MagicMock()
        )
        with self.assertRaises(ValueError) as cm:
            GenAIPipeline.from_proxy(proxy)
        self.assertIn("No inference strategy", str(cm.exception))


class TestGenAIPipelineInvoke(unittest.TestCase):

    def test_invoke_with_mock_strategies(self):
        proxy = EngineProxy(
            {
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
                STAGE_INFERENCE: EngineType.EXECUTORCH,
            },
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=QuantizationStage(_MockQuantizationStrategy()),
            compilation_stage=CompilationStage(_MockCompilationStrategy()),
            inference_stage=InferenceStage(_MockInferenceStrategy()),
            engine_proxy=proxy,
        )
        result = pipeline.invoke(make_test_context())

        self.assertIsInstance(result, InferenceOutputConfig)
        method_result = result.results[TEST_EVAL_METHOD]
        self.assertEqual(method_result.generations, [TEST_MOCK_GENERATED_TEXT])
        self.assertEqual(
            method_result.performance_metrics["tokens_per_sec"],
            TEST_MOCK_TOKENS_PER_SEC,
        )

    def test_invoke_compile_only(self):
        proxy = EngineProxy(
            {
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
            },
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=QuantizationStage(_MockQuantizationStrategy()),
            compilation_stage=CompilationStage(_MockCompilationStrategy()),
            inference_stage=None,
            engine_proxy=proxy,
        )
        result = pipeline.invoke(make_test_context())

        self.assertIsInstance(result, InferenceOutputConfig)

    def test_invoke_no_stages(self):
        proxy = EngineProxy({}, backend_type=TEST_BACKEND_TYPE)
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=None,
            compilation_stage=None,
            inference_stage=None,
            engine_proxy=proxy,
        )
        result = pipeline.invoke(make_test_context())

        self.assertIsInstance(result, InferenceOutputConfig)

    def test_quantization_receives_soc_model(self):
        mock_quant = MagicMock(spec=QuantizationStrategy)
        mock_quant.invoke.return_value = QuantizationOutputConfig(graphs={})

        test_soc = "SM8650"
        proxy = EngineProxy(
            {STAGE_QUANTIZATION: EngineType.EXECUTORCH},
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=QuantizationStage(mock_quant),
            compilation_stage=None,
            inference_stage=None,
            engine_proxy=proxy,
        )
        pipeline.invoke(make_test_context(soc_model=test_soc))

        args, _ = mock_quant.invoke.call_args
        input_config = args[1]
        self.assertEqual(input_config.soc_model, test_soc)

    def test_model_preparation_receives_full_extra_options(self):
        mock_model_prep = MagicMock(spec=ModelPreparationStrategy)
        mock_model_prep.invoke.return_value = ModelPreparationOutputConfig()
        extra_options = {
            "model_options": {"model_arch": MagicMock(name="model_arch")},
            "quantize_options": {"quant_dtype": MagicMock(name="quant_dtype")},
            "dataset_options": MagicMock(name="dataset_options"),
        }
        proxy = EngineProxy(
            {STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH},
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=ModelPreparationStage(mock_model_prep),
            quantization_stage=None,
            compilation_stage=None,
            inference_stage=None,
            engine_proxy=proxy,
        )

        pipeline.invoke(make_test_context(extra_options=extra_options))

        args, _ = mock_model_prep.invoke.call_args
        input_config = args[1]
        self.assertEqual(input_config.extra_options, extra_options)

    def test_quantization_receives_full_extra_options(self):
        mock_quant = MagicMock(spec=QuantizationStrategy)
        mock_quant.invoke.return_value = QuantizationOutputConfig(graphs={})
        extra_options = {
            "model_options": {"model_arch": MagicMock(name="model_arch")},
            "quantize_options": {"quant_dtype": MagicMock(name="quant_dtype")},
            "dataset_options": MagicMock(name="dataset_options"),
        }
        proxy = EngineProxy(
            {STAGE_QUANTIZATION: EngineType.EXECUTORCH},
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=QuantizationStage(mock_quant),
            compilation_stage=None,
            inference_stage=None,
            engine_proxy=proxy,
        )

        pipeline.invoke(make_test_context(extra_options=extra_options))

        args, _ = mock_quant.invoke.call_args
        input_config = args[1]
        self.assertEqual(input_config.extra_options, extra_options)

    def test_compilation_receives_backend_type(self):
        mock_compile = MagicMock(spec=CompilationStrategy)
        mock_compile.invoke.return_value = CompilationOutputConfig(
            artifact_paths=TEST_ARTIFACT_PATHS
        )

        mock_backend_type = MagicMock()
        proxy = EngineProxy(
            {STAGE_COMPILATION: EngineType.EXECUTORCH},
            backend_type=mock_backend_type,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=None,
            compilation_stage=CompilationStage(mock_compile),
            inference_stage=None,
            engine_proxy=proxy,
        )
        pipeline.invoke(make_test_context())

        call_args = mock_compile.invoke.call_args
        input_config = call_args[0][1]
        self.assertEqual(input_config.backend_type, mock_backend_type)

    def test_compilation_receives_model_preparation_outputs(self):
        """Without quantization, compilation gets the prepared model as is."""
        from executorch.backends.qualcomm.genai_pipeline.compilation.model_layout import (
            ModelLayout,
        )

        layout = ModelLayout(decoder_layer_prefixes=("model.layers",))
        prep_output = ModelPreparationOutputConfig(
            model_module={"text_decoder": MagicMock(name="decoder")},
            example_inputs={"text_decoder": {"kv_forward": ("tokens",)}},
            meta={"text_decoder": {"kv_forward": {"get_n_layers": 2}}},
            num_shardings=None,
            model_layout=layout,
        )
        mock_prep = MagicMock(spec=ModelPreparationStrategy)
        mock_prep.invoke.return_value = prep_output
        mock_compile = MagicMock(spec=CompilationStrategy)
        mock_compile.invoke.return_value = CompilationOutputConfig(
            artifact_paths=TEST_ARTIFACT_PATHS
        )
        proxy = EngineProxy(
            {
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
            },
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=ModelPreparationStage(mock_prep),
            quantization_stage=None,
            compilation_stage=CompilationStage(mock_compile),
            inference_stage=None,
            engine_proxy=proxy,
        )

        pipeline.invoke(make_test_context())

        input_config = mock_compile.invoke.call_args[0][1]
        self.assertIsNone(input_config.graphs)
        self.assertIs(input_config.model, prep_output.model_module)
        self.assertIs(input_config.example_inputs, prep_output.example_inputs)
        self.assertIs(input_config.meta, prep_output.meta)
        self.assertEqual(input_config.num_sharding, 1)
        self.assertIs(input_config.model_layout, layout)

    def test_inference_receives_prompt(self):
        mock_infer = MagicMock(spec=InferenceStrategy)
        mock_infer.invoke.return_value = InferenceOutputConfig()

        test_prompt = ["What is AI?"]
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH},
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=None,
            quantization_stage=None,
            compilation_stage=None,
            inference_stage=InferenceStage(mock_infer),
            engine_proxy=proxy,
        )
        pipeline.invoke(make_test_context(prompt=test_prompt))

        call_args = mock_infer.invoke.call_args
        input_config = call_args[0][1]
        self.assertEqual(input_config.prompt, test_prompt)

    def _invoke_inference(self, extra_options, model_prep_output):
        """Run a pipeline of model preparation and inference, returning the inference input."""
        mock_model_prep = MagicMock(spec=ModelPreparationStrategy)
        mock_model_prep.invoke.return_value = model_prep_output
        mock_infer = MagicMock(spec=InferenceStrategy)
        mock_infer.invoke.return_value = InferenceOutputConfig()
        proxy = EngineProxy(
            {
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_INFERENCE: EngineType.EXECUTORCH,
            },
            backend_type=TEST_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline(
            model_preparation_stage=ModelPreparationStage(mock_model_prep),
            quantization_stage=None,
            compilation_stage=None,
            inference_stage=InferenceStage(mock_infer),
            engine_proxy=proxy,
        )
        pipeline.invoke(make_test_context(extra_options=extra_options))
        return mock_infer.invoke.call_args[0][1]

    def test_inference_receives_model_preparation_runtime_tokenizer(self):
        """The runtime tokenizer passes through as model preparation reported it."""
        prep_tokenizer = Path("/tmp/prep/tokenizer.json")
        input_config = self._invoke_inference(
            extra_options={
                "runtime_tokenizer_path": "/tmp/override.bin",
                "inference_options": {"runtime_tokenizer_path": "/tmp/nested.bin"},
            },
            model_prep_output=ModelPreparationOutputConfig(
                runtime_tokenizer_path=prep_tokenizer
            ),
        )

        self.assertEqual(input_config.runtime_tokenizer_path, prep_tokenizer)

    def test_inference_receives_inference_options(self):
        """extra_options['inference_options'] is forwarded to the inference input."""
        inference_options = {"temperature": 0.5, "system_prompt": "be brief"}
        input_config = self._invoke_inference(
            extra_options={"inference_options": inference_options},
            model_prep_output=ModelPreparationOutputConfig(),
        )

        self.assertEqual(input_config.inference_options, inference_options)

    def test_inference_options_default_to_empty(self):
        """Without inference_options the inference input carries an empty dict."""
        input_config = self._invoke_inference(
            extra_options={},
            model_prep_output=ModelPreparationOutputConfig(),
        )

        self.assertEqual(input_config.inference_options, {})


if __name__ == "__main__":
    unittest.main()
