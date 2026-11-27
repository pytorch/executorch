# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


class CompilationStrategy(ABC):
    @abstractmethod
    def invoke(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
    ) -> CompilationOutputConfig:
        """Compile the model to on-device artifacts.

        Args:
            context: The pipeline context with global settings.
            input_config: The compilation input configuration.

        Returns:
            CompilationOutputConfig with artifact paths.
        """
        ...
