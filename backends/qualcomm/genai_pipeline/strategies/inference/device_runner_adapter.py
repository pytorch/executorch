# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class InferenceResult:
    """Result of an on-device inference run.

    Attributes:
        output_data: Raw output data from the model execution.
        performance_metrics: Performance data (e.g., TTFT, tokens/sec).
        etdump: Optional ETDump for debugging.
    """

    output_data: Optional[List[Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    etdump: Optional[Any] = None


@runtime_checkable
class DeviceRunnerAdapter(Protocol):
    """Protocol for on-device inference operations.

    Wraps external device runner APIs (SimpleADB) behind an injectable
    interface for testability.
    """

    def push_artifacts(
        self,
        artifact_paths: List[Path],
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Push compiled artifacts and inputs to the device.

        Args:
            artifact_paths: Paths to compiled .pte artifacts.
            input_data: Optional input data to push to device.
            extra_files: Optional additional files to push.
        """
        ...

    def execute(
        self,
        inference_options: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Execute the model on device.

        Args:
            inference_options: Engine-specific inference options.

        Returns:
            InferenceResult with output data and performance metrics.
        """
        ...

    def pull_results(
        self,
        output_dir: Path,
    ) -> List[Path]:
        """Pull inference results from the device.

        Args:
            output_dir: Local directory to store pulled results.

        Returns:
            List of paths to pulled result files.
        """
        ...
