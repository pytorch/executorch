# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "1.0.0"

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ALL_ARTIFACT_KEYS,
    ARTIFACT_ATTENTION_SINK_EVICTOR,
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
    DECODE_QDQ_FILENAME,
)
from executorch.backends.qualcomm.genai_pipeline.compilation import (
    QnnCompileSpecBuilder,
    resolve_backend_type,
    resolve_soc_model,
)
from executorch.backends.qualcomm.genai_pipeline.configs import (
    CompilationInputConfig,
    CompilationOutputConfig,
    InferenceInputConfig,
    InferenceOutputConfig,
    ModelPreparationInputConfig,
    ModelPreparationOutputConfig,
    QuantizationInputConfig,
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
from executorch.backends.qualcomm.genai_pipeline.exceptions import (
    ConfigValidationError,
    EngineNotAvailableError,
    PipelineError,
    StageError,
)
from executorch.backends.qualcomm.genai_pipeline.genai_pipeline import GenAIPipeline
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
    GRAPH_PREFILL_FORWARD,
    GRAPH_TOK_EMBEDDING_KV_FORWARD,
    GRAPH_TOK_EMBEDDING_PREFILL_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import (
    PipelineContext,
    PipelineContextBuilder,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_stage import PipelineStage
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import EngineType

__all__ = [
    "ALL_ARTIFACT_KEYS",
    "ARTIFACT_ATTENTION_SINK_EVICTOR",
    "ARTIFACT_AUDIO_ENCODER",
    "ARTIFACT_TEXT_DECODER",
    "ARTIFACT_TEXT_ENCODER",
    "ARTIFACT_TOK_EMBEDDING",
    "ARTIFACT_VISION_ENCODER",
    "CompilationInputConfig",
    "CompilationOutputConfig",
    "ConfigValidationError",
    "ControlArgs",
    "DECODER_GRAPH_NAMES",
    "DECODE_QDQ_FILENAME",
    "EngineNotAvailableError",
    "EngineProxy",
    "EngineType",
    "GenAIPipeline",
    "GRAPH_FORWARD",
    "GRAPH_KV_FORWARD",
    "GRAPH_PREFILL_FORWARD",
    "GRAPH_TOK_EMBEDDING_KV_FORWARD",
    "GRAPH_TOK_EMBEDDING_PREFILL_FORWARD",
    "GraphBundle",
    "InferenceInputConfig",
    "InferenceOutputConfig",
    "ModelPreparationInputConfig",
    "ModelPreparationOutputConfig",
    "PipelineContext",
    "PipelineContextBuilder",
    "PipelineError",
    "PipelineStage",
    "QnnCompileSpecBuilder",
    "QuantizationInputConfig",
    "QuantizationOutputConfig",
    "resolve_backend_type",
    "resolve_soc_model",
    "StageError",
    "TOK_EMBEDDING_GRAPH_NAMES",
]
