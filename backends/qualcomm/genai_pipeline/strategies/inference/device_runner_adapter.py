# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Protocol, runtime_checkable

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    RunnerMetadata,
    RunnerOutput,
    RunnerRequest,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


@runtime_checkable
class DeviceRunnerAdapter(Protocol):
    """Lifecycle of a compiled model on a runner.

    The methods are lifecycle stages rather than transfer operations:
    ``push_artifacts`` deploys the artifacts where the runner can read them,
    ``get_metadata`` reports the deployed model's metadata, ``push_inputs``
    makes one request's inputs readable by the runner, ``execute`` runs that
    request, and ``pull_results`` collects its raw output. The per-request
    stages repeat, in that order, after a single ``push_artifacts``. A host-side
    runner implements the same stages without any file transfer.
    """

    def push_artifacts(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Deploy ``input_config.pte_paths`` and runtime files.

        Args:
            context: Pipeline context with the model identity and control args.
            input_config: Inference configuration with the PTE paths, artifact
                directory, and runtime tokenizer.
            input_data: Pre-encoded inputs to deploy with the artifacts, one
                positional-input tuple per sample, for a caller that drives a
                generic runner directly. The strategy passes none: evaluators
                supply their inputs through their requests.
            extra_files: Additional files to deploy alongside the artifacts.
        """
        ...

    def get_metadata(self) -> Optional[RunnerMetadata]:
        """Return the deployed decoder's metadata, or ``None`` if not read."""
        ...

    def push_inputs(self, request: RunnerRequest) -> None:
        """Make the inputs ``request`` carries readable by the runner."""
        ...

    def execute(self, request: RunnerRequest) -> None:
        """Run one request whose inputs ``push_inputs`` made readable."""
        ...

    def pull_results(self, request: RunnerRequest, output_dir: Path) -> RunnerOutput:
        """Collect the raw output of ``request``, which ``execute`` just ran."""
        ...
