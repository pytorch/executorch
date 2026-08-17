# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from executorch.backends.qualcomm.genai_pipeline.strategies.inference.device_runner_adapter import (
    InferenceResult,
)

logger = logging.getLogger(__name__)


class DefaultDeviceRunnerAdapter:
    """Default adapter delegating to ``SimpleADB`` for on-device inference.

    Wraps the ``SimpleADB`` class from ``export_utils.py`` to push artifacts,
    execute models, and pull results from an Android device via ADB.

    Args:
        qnn_config: A ``QnnConfig`` instance for device communication.
        workspace: Folder for storing artifacts on the Android device.
    """

    def __init__(
        self,
        qnn_config: Any,
        workspace: str = "/data/local/tmp/genai_pipeline",
    ) -> None:
        self._qnn_config = qnn_config
        self._workspace = workspace
        self._adb: Any = None

    def push_artifacts(
        self,
        artifact_paths: List[Path],
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Push compiled artifacts and inputs to the device via ADB.

        Args:
            artifact_paths: Paths to compiled .pte artifacts.
            input_data: Optional input data to push to device.
            extra_files: Optional additional files to push.
        """
        from executorch.backends.qualcomm.export_utils import SimpleADB

        if self._adb is not None:
            logger.warning(
                "Overwriting existing ADB session. Previous artifacts will be "
                "replaced on device."
            )

        pte_paths = [str(p) for p in artifact_paths]

        logger.info("Pushing artifacts to device: %s", pte_paths)
        self._adb = SimpleADB(
            qnn_config=self._qnn_config,
            pte_path=pte_paths,
            workspace=self._workspace,
        )
        self._adb.push(
            inputs=input_data,
            files=extra_files,
        )

    def execute(
        self,
        inference_options: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Execute the model on device via ADB.

        Args:
            inference_options: Engine-specific options. Supported keys:
                - ``method_index``: Index of the method to execute (default 0).
                - ``iteration``: Number of inference iterations (default 1).

        Returns:
            InferenceResult with output data and performance metrics.
        """
        if self._adb is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        inference_options = inference_options or {}
        method_index = inference_options.get("method_index", 0)
        iteration = inference_options.get("iteration", 1)

        logger.info(
            "Executing on device: method_index=%d, iteration=%d",
            method_index,
            iteration,
        )
        self._adb.execute(
            method_index=method_index,
            iteration=iteration,
        )

        return InferenceResult(
            output_data=None,
            performance_metrics={
                "method_index": method_index,
                "iteration": iteration,
            },
        )

    def pull_results(
        self,
        output_dir: Path,
    ) -> List[Path]:
        """Pull inference results from the device via ADB.

        Args:
            output_dir: Local directory to store pulled results.

        Returns:
            List of paths to pulled result files.
        """
        if self._adb is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pulling results from device to %s", output_dir)
        self._adb.pull(host_output_path=str(output_dir))

        # adb pull of a directory creates a subdirectory locally;
        # collect actual result files from pulled content
        result_files = []
        for item in output_dir.iterdir():
            if item.is_dir():
                result_files.extend(item.iterdir())
            else:
                result_files.append(item)
        logger.info("Pulled %d result files", len(result_files))
        return result_files
