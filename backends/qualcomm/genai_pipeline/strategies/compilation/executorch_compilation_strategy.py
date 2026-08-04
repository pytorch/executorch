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
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)


class ExecuTorchCompilationStrategy(CompilationStrategy):
    """ExecuTorch-based compilation using QNN compiler backend.

    Reads backend_type from the input config to select the appropriate compiler backend (HTP, GPU, or LPAI).
    """

    def invoke(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
    ) -> CompilationOutputConfig:
        raise NotImplementedError(
            "ExecuTorchCompilationStrategy.invoke() is a stub. "
            "Implementation will be added in a subsequent PR."
        )
