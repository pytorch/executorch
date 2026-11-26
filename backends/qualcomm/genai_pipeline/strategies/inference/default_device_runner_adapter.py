# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import getpass
import logging
from pathlib import Path
from typing import Any, List, Optional

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteOutput,
    ExecuteRequest,
)

logger = logging.getLogger(__name__)


class DefaultDeviceRunnerAdapter:
    """Default adapter delegating to ``Device`` for on-device inference.

    Implements the ``DeviceRunnerAdapter`` lifecycle for the generic executor
    runner on a target device: a remote Android/OE-Linux device over adb, or a
    Windows-on-Snapdragon device the CLI runs on locally, as ``Device`` picks
    from ``qnn_config.target``.

    Unlike the LLM and MLLM runner adapters, it drives ``Device`` directly
    rather than through a ``RunnerPlatform``: it runs ``qnn_executor_runner``
    through ``Device``'s own default command, which a platform does not model.

    The runner reads the input list ``push_inputs`` deploys from the request's
    data samples, which ``DefaultEvaluator`` is given::

        ExecuTorchInferenceStrategy(
            device_runner_adapter=DefaultDeviceRunnerAdapter(qnn_config),
            evaluators=[DefaultEvaluator(inputs=dataset_samples)],
        )

    The CLI does not build this path: it serves only registry LLMs.

    Args:
        qnn_config: A ``QnnConfig`` instance for device communication.
        workspace: Folder for storing artifacts on the target device. Defaults
            to a per-user folder, as the legacy flow does: ``push_artifacts``
            clears the workspace, so a shared one would wipe another user's or
            CI job's files mid-run.
    """

    def __init__(
        self,
        qnn_config: Any,
        workspace: Optional[str] = None,
    ) -> None:
        self._qnn_config = qnn_config
        self._workspace = (
            workspace or f"/data/local/tmp/{getpass.getuser()}/genai_pipeline"
        )
        self._device: Any = None

    def push_artifacts(
        self,
        control_args: ControlArgs,
        input_config: InferenceInputConfig,
        input_data: Optional[List[Any]] = None,
        extra_files: Optional[List[str]] = None,
    ) -> None:
        """Push compiled artifacts and inputs to the target device.

        Args:
            control_args: Unused; the generic runner takes no control args.
            input_config: Inference configuration whose ``pte_paths`` are all
                pushed; the keys select which on-device runner argument each
                path is passed to, which is the runner's concern rather than
                this adapter's.
            input_data: Inputs to deploy up front, one positional-input tuple
                per sample, for a caller that drives this adapter directly;
                requests otherwise bring their samples through ``push_inputs``.
            extra_files: Optional additional files to push.
        """
        from executorch.backends.qualcomm.export_utils import Device

        if self._device is not None:
            logger.warning(
                "Overwriting existing device session. Previous artifacts will be "
                "replaced on device."
            )

        pte_paths = [str(p) for p in input_config.pte_paths.values()]

        logger.info("Pushing artifacts to device: %s", pte_paths)
        self._device = Device(
            qnn_config=self._qnn_config,
            pte_path=pte_paths,
            workspace=self._workspace,
        )
        self._device.push(
            inputs=input_data,
            files=extra_files,
        )

    def get_metadata(self) -> None:
        """Return ``None``: the generic executor runner has no decoder metadata."""
        return None

    def push_inputs(self, request: ExecuteRequest) -> None:
        """Push the request's samples as the runner's input list.

        Args:
            request: The samples to run.
        """
        if self._device is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )
        # Device writes the samples as the input list its default command reads.
        self._device.push(inputs=list(request.inputs), init_env=False)

    def execute(self, request: ExecuteRequest) -> None:
        """Execute one compiled method on the target device, once over all samples.

        Args:
            request: The method index and iteration count to run.
        """
        if self._device is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )
        logger.info(
            "Executing on device: method_index=%d, iteration=%d",
            request.method_index,
            request.iteration,
        )
        self._device.execute(
            method_index=request.method_index,
            iteration=request.iteration,
        )

    def pull_results(self, request: ExecuteRequest, output_dir: Path) -> ExecuteOutput:
        """Pull the device output folder of the request ``execute`` just ran.

        Args:
            request: The request ``execute`` just ran.
            output_dir: Local directory to store pulled results.

        Returns:
            The pulled output files.
        """
        if self._device is None:
            raise RuntimeError(
                "No artifacts have been pushed. Call push_artifacts() first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pulling results from device to %s", output_dir)
        self._device.pull(host_output_path=str(output_dir))

        # pulling a directory creates a subdirectory locally;
        # collect actual result files from pulled content
        result_files = []
        for item in output_dir.iterdir():
            if item.is_dir():
                result_files.extend(item.iterdir())
            else:
                result_files.append(item)
        logger.info("Pulled %d result files", len(result_files))
        return ExecuteOutput(files=result_files)
