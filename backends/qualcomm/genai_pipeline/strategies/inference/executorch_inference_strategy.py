# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.device_runner_adapter import (
    DeviceRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "inference"


class ExecuTorchInferenceStrategy(InferenceStrategy):
    """ExecuTorch-based inference using QNN runtime on device.

    Delegates to a ``DeviceRunnerAdapter`` for all external API calls,
    enabling dependency injection for testability.

    The inference flow:
    1. Validate the input configuration
    2. Push artifacts to device
    3. Execute the model on device
    4. Pull and return results

    Input preparation belongs to the adapter, not here. ``DeviceRunnerAdapter``
    accepts ``input_data`` / ``extra_files`` for adapters that prefer the caller
    to supply pre-encoded inputs, but this strategy passes only the artifacts:
    turning ``prompt`` into model inputs needs the tokenizer's chat template,
    BOS handling, AR-length padding and KV-cache seeding, all of which the
    adapter knows and the strategy does not. Decoding is the adapter's for the
    same reason, so ``InferenceResult.output_data`` arrives as text and is
    forwarded without conversion.

    Args:
        device_runner_adapter: Injectable adapter for device inference operations.
            Must be provided (no default, since device configuration is required).
    """

    def __init__(
        self,
        device_runner_adapter: Optional[DeviceRunnerAdapter] = None,
    ) -> None:
        self._adapter = device_runner_adapter

    @property
    def adapter(self) -> Optional[DeviceRunnerAdapter]:
        """The device runner adapter used by this strategy."""
        return self._adapter

    def invoke(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceOutputConfig:
        """Run inference using compiled model artifacts on device.

        Args:
            context: The pipeline context with global settings.
            input_config: The inference input configuration.

        Returns:
            InferenceOutputConfig with inference results and metrics.

        Raises:
            StageError: If artifacts are missing, adapter is not configured,
                or inference fails.
        """
        self._validate_input(input_config)

        logger.info(
            "Starting inference for model '%s' on SoC=%s",
            context.model_name,
            input_config.soc_model,
        )

        try:
            # Step 1: Push artifacts to device. Only the artifacts: the adapter
            # owns prompt -> tokens -> device inputs (see class docstring), so
            # input_data/extra_files are deliberately not passed here.
            logger.debug("Pushing artifacts to device")
            self._adapter.push_artifacts(
                artifact_paths=input_config.artifact_paths,
            )

            # Step 2: Execute on device
            logger.debug("Executing model on device")
            result = self._adapter.execute(
                inference_options=input_config.inference_options,
            )

            # Step 3: Pull results from device
            output_dir = Path(context.artifact_dir) / "inference_output"
            logger.debug("Pulling results to %s", output_dir)
            result_files = self._adapter.pull_results(output_dir=output_dir)

            logger.info("Inference completed successfully")

            # ``output_data`` is already decoded text by contract (the adapter
            # owns decoding), so it is forwarded as-is; only the pulled-file
            # fallback needs converting, because those genuinely are paths.
            inference_results = result.output_data
            if inference_results is None and result_files:
                inference_results = [str(p) for p in result_files]

            return InferenceOutputConfig(
                inference_results=inference_results,
                performance_metrics=result.performance_metrics,
                etdump=result.etdump,
            )

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Inference failed",
                original_exception=e,
            ) from e

    def _validate_input(self, input_config: InferenceInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The inference input configuration.

        Raises:
            StageError: If required fields are missing or adapter is not set.
        """
        if self._adapter is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="device_runner_adapter is required for inference. "
                "Provide a DeviceRunnerAdapter via the constructor.",
            )
        if not input_config.artifact_paths:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="artifact_paths is required for inference",
            )
