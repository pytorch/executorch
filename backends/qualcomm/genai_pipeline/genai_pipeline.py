# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Type

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
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
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.executorch_compilation_strategy import (
    ExecuTorchCompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
    ExecuTorchInferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.executorch_model_preparation_strategy import (
    ExecuTorchModelPreparationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
    ExecuTorchQuantizationStrategy,
)

logger = logging.getLogger("genai_pipeline")

# Maps (stage_name, engine_type) → (StageClass, StrategyClass)
_STRATEGY_REGISTRY: Dict[Tuple[str, EngineType], Tuple[Type, Type]] = {
    (STAGE_MODEL_PREPARATION, EngineType.EXECUTORCH): (
        ModelPreparationStage,
        ExecuTorchModelPreparationStrategy,
    ),
    (STAGE_QUANTIZATION, EngineType.EXECUTORCH): (
        QuantizationStage,
        ExecuTorchQuantizationStrategy,
    ),
    (STAGE_COMPILATION, EngineType.EXECUTORCH): (
        CompilationStage,
        ExecuTorchCompilationStrategy,
    ),
    (STAGE_INFERENCE, EngineType.EXECUTORCH): (
        InferenceStage,
        ExecuTorchInferenceStrategy,
    ),
}


class GenAIPipeline:
    """Main pipeline orchestrator for GenAI model workflows.

    Assembles stages from EngineProxy, wires data flow between
    InputConfig → Stage → OutputConfig, and executes sequentially:
    model_preparation → quantization → compilation → inference.
    """

    def __init__(
        self,
        model_preparation_stage: Optional[ModelPreparationStage],
        quantization_stage: Optional[QuantizationStage],
        compilation_stage: Optional[CompilationStage],
        inference_stage: Optional[InferenceStage],
        engine_proxy: EngineProxy,
    ) -> None:
        self._model_preparation_stage = model_preparation_stage
        self._quantization_stage = quantization_stage
        self._compilation_stage = compilation_stage
        self._inference_stage = inference_stage
        self._engine_proxy = engine_proxy

    @classmethod
    def from_proxy(
        cls,
        engine_proxy: EngineProxy,
        skip_stages: Optional[Set[str]] = None,
    ) -> "GenAIPipeline":
        """Create a pipeline from an EngineProxy configuration.

        Args:
            engine_proxy: The EngineProxy with per-stage engine assignments.
            skip_stages: Optional set of stage names to skip.

        Returns:
            A fully configured GenAIPipeline.
        """
        skip = skip_stages or set()

        model_prep_stage = cls._resolve_stage(
            engine_proxy, STAGE_MODEL_PREPARATION, skip
        )
        quant_stage = cls._resolve_stage(engine_proxy, STAGE_QUANTIZATION, skip)
        compile_stage = cls._resolve_stage(engine_proxy, STAGE_COMPILATION, skip)
        infer_stage = cls._resolve_stage(engine_proxy, STAGE_INFERENCE, skip)

        return cls(
            model_preparation_stage=model_prep_stage,
            quantization_stage=quant_stage,
            compilation_stage=compile_stage,
            inference_stage=infer_stage,
            engine_proxy=engine_proxy,
        )

    @staticmethod
    def _resolve_stage(
        engine_proxy: EngineProxy,
        stage_name: str,
        skip: set,
    ):
        if stage_name in skip:
            return None

        engine = engine_proxy.get_engine(stage_name)
        key = (stage_name, engine)

        if key not in _STRATEGY_REGISTRY:
            raise ValueError(f"No {stage_name} strategy for engine: {engine}")

        stage_cls, strategy_cls = _STRATEGY_REGISTRY[key]
        return stage_cls(strategy_cls())

    def invoke(self, context: PipelineContext) -> InferenceOutputConfig:
        """Execute the full pipeline sequentially.

        Args:
            context: The pipeline context with user inputs.

        Returns:
            InferenceOutputConfig with inference results and metrics.
        """
        logger.info(
            "[GenAIPipeline] Pipeline started for model '%s'", context.model_name
        )

        model_prep_output = self._run_model_preparation(context)
        quant_output = self._run_quantization(context, model_prep_output)
        compile_output = self._run_compilation(context, quant_output)
        result = self._run_inference(context, model_prep_output, compile_output)

        logger.info("[GenAIPipeline] Pipeline completed")
        return result

    def _run_model_preparation(
        self, context: PipelineContext
    ) -> ModelPreparationOutputConfig:
        if self._model_preparation_stage is not None:
            logger.info("[GenAIPipeline] ModelPreparationStage started")
            start = time.monotonic()
            input_config = ModelPreparationInputConfig(
                model_name=context.model_name,
                soc_model=context.soc_model,
            )
            output = self._model_preparation_stage.invoke(context, input_config)
            elapsed = time.monotonic() - start
            logger.info(
                "[GenAIPipeline] ModelPreparationStage completed in %.1fs", elapsed
            )
            return output
        return ModelPreparationOutputConfig()

    def _run_quantization(
        self,
        context: PipelineContext,
        model_prep_output: ModelPreparationOutputConfig,
    ) -> QuantizationOutputConfig:
        if self._quantization_stage is not None:
            logger.info("[GenAIPipeline] QuantizationStage started")
            start = time.monotonic()
            input_config = QuantizationInputConfig(
                soc_model=context.soc_model,
                backend_type=self._engine_proxy.backend_type,
                model_module=model_prep_output.model_module,
                calibration_data=model_prep_output.calibration_data,
            )
            output = self._quantization_stage.invoke(context, input_config)
            elapsed = time.monotonic() - start
            logger.info("[GenAIPipeline] QuantizationStage completed in %.1fs", elapsed)
            return output
        return QuantizationOutputConfig()

    def _run_compilation(
        self,
        context: PipelineContext,
        quant_output: QuantizationOutputConfig,
    ) -> CompilationOutputConfig:
        if self._compilation_stage is not None:
            logger.info("[GenAIPipeline] CompilationStage started")
            start = time.monotonic()
            input_config = CompilationInputConfig(
                soc_model=context.soc_model,
                backend_type=self._engine_proxy.backend_type,
                model=quant_output.quantized_model,
                artifact_dir=Path(context.artifact_dir),
            )
            output = self._compilation_stage.invoke(context, input_config)
            elapsed = time.monotonic() - start
            logger.info("[GenAIPipeline] CompilationStage completed in %.1fs", elapsed)
            return output
        return CompilationOutputConfig()

    def _run_inference(
        self,
        context: PipelineContext,
        model_prep_output: ModelPreparationOutputConfig,
        compile_output: CompilationOutputConfig,
    ) -> InferenceOutputConfig:
        if self._inference_stage is not None:
            logger.info("[GenAIPipeline] InferenceStage started")
            start = time.monotonic()
            input_config = InferenceInputConfig(
                artifact_paths=compile_output.artifact_paths,
                tokenizer=model_prep_output.tokenizer,
                runtime_tokenizer_path=model_prep_output.runtime_tokenizer_path,
                prompt=context.prompt,
                soc_model=context.soc_model,
            )
            output = self._inference_stage.invoke(context, input_config)
            elapsed = time.monotonic() - start
            logger.info("[GenAIPipeline] InferenceStage completed in %.1fs", elapsed)
            return output
        return InferenceOutputConfig()
