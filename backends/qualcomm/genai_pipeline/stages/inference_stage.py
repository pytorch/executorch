# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.pipeline_stage import PipelineStage
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import STAGE_INFERENCE
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)


class InferenceStage(PipelineStage):

    def __init__(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        return STAGE_INFERENCE

    def invoke(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceOutputConfig:
        return self._strategy.invoke(context, input_config)
