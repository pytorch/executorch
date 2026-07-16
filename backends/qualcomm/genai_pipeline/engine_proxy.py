# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    ALL_STAGES,
    EngineType,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )


class EngineProxy:
    """Routes each pipeline stage to the appropriate engine strategy.

    Validates stage-engine combinations at construction time.

    Args:
        stage_engines: Mapping of stage name to EngineType.
        backend_type: The QNN backend type (HTP, GPU, LPAI, etc.).

    Raises:
        ValueError: If an unsupported stage-engine combination is specified.
    """

    def __init__(
        self,
        stage_engines: Dict[str, EngineType],
        backend_type: "QnnExecuTorchBackendType",
    ):
        self._validate(stage_engines)
        self._stage_engines = dict(stage_engines)
        self._backend_type = backend_type

    def _validate(
        self,
        stage_engines: Dict[str, EngineType],
    ) -> None:
        """Validate stage-engine combinations.

        Raises:
            ValueError: On invalid stage names.
        """
        for stage in stage_engines:
            if stage not in ALL_STAGES:
                raise ValueError(
                    f"Unknown stage '{stage}'. " f"Valid stages: {sorted(ALL_STAGES)}"
                )

    def get_engine(self, stage: str) -> EngineType:
        """Get the engine type for a given stage.

        Args:
            stage: The pipeline stage name.

        Returns:
            The EngineType for the stage, defaulting to EXECUTORCH
            if not explicitly configured.
        """
        return self._stage_engines.get(stage, EngineType.EXECUTORCH)

    @property
    def backend_type(self) -> "QnnExecuTorchBackendType":
        return self._backend_type

    @property
    def stage_engines(self) -> Dict[str, EngineType]:
        return dict(self._stage_engines)
