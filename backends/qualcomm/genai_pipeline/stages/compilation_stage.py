# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.pipeline_stage import PipelineStage
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import STAGE_COMPILATION
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)


class CompilationStage(PipelineStage):

    def __init__(self, strategy: CompilationStrategy) -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        return STAGE_COMPILATION

    def invoke(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
    ) -> CompilationOutputConfig:
        return self._strategy.invoke(context, input_config)
