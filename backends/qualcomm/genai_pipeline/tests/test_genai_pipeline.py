# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
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
from executorch.backends.qualcomm.genai_pipeline.stages.quantization_stage import (
    QuantizationStage,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    make_test_context,
    TEST_PTE_PATH,
)

TEST_MOCK_GENERATED_TEXT = "Mock generated text"
TEST_MOCK_TOKENS_PER_SEC = 42.0
TEST_MOCK_BACKEND_TYPE = MagicMock(name="kHtpBackend")


class _MockQuantizationStrategy(QuantizationStrategy):
    def invoke(self, context, input_config):
        return QuantizationOutputConfig(quantized_model="mock_quantized_model")


class _MockCompilationStrategy(CompilationStrategy):
    def invoke(self, context, input_config):
        return CompilationOutputConfig(artifact_paths=[TEST_PTE_PATH])


class _MockInferenceStrategy(InferenceStrategy):
    def invoke(self, context, input_config):
        return InferenceOutputConfig(
            inference_results=[TEST_MOCK_GENERATED_TEXT],
            performance_metrics={"tokens_per_sec": TEST_MOCK_TOKENS_PER_SEC},
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
            backend_type=TEST_MOCK_BACKEND_TYPE,
        )
        pipeline = GenAIPipeline.from_proxy(proxy)
        self.assertIsNotNone(pipeline._model_preparation_stage)
        self.assertIsNotNone(pipeline._quantization_stage)
        self.assertIsNotNone(pipeline._compilation_stage)
        self.assertIsNotNone(pipeline._inference_stage)

    def test_skip_stages(self):
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH},
            backend_type=TEST_MOCK_BACKEND_TYPE,
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
        proxy = EngineProxy({}, backend_type=TEST_MOCK_BACKEND_TYPE)
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
            backend_type=TEST_MOCK_BACKEND_TYPE,
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
        self.assertEqual(result.inference_results, [TEST_MOCK_GENERATED_TEXT])
        self.assertEqual(
            result.performance_metrics["tokens_per_sec"], TEST_MOCK_TOKENS_PER_SEC
        )

    def test_invoke_compile_only(self):
        proxy = EngineProxy(
            {
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
            },
            backend_type=TEST_MOCK_BACKEND_TYPE,
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
        proxy = EngineProxy({}, backend_type=TEST_MOCK_BACKEND_TYPE)
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
        mock_quant.invoke.return_value = QuantizationOutputConfig(
            quantized_model="quantized"
        )

        test_soc = "SM8650"
        proxy = EngineProxy(
            {STAGE_QUANTIZATION: EngineType.EXECUTORCH},
            backend_type=TEST_MOCK_BACKEND_TYPE,
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

    def test_compilation_receives_backend_type(self):
        mock_compile = MagicMock(spec=CompilationStrategy)
        mock_compile.invoke.return_value = CompilationOutputConfig(
            artifact_paths=[TEST_PTE_PATH]
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

    def test_inference_receives_prompt(self):
        mock_infer = MagicMock(spec=InferenceStrategy)
        mock_infer.invoke.return_value = InferenceOutputConfig(
            inference_results=["output"]
        )

        test_prompt = ["What is AI?"]
        proxy = EngineProxy(
            {STAGE_INFERENCE: EngineType.EXECUTORCH},
            backend_type=TEST_MOCK_BACKEND_TYPE,
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


if __name__ == "__main__":
    unittest.main()
