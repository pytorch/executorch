# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.calibration_data_adapter import (
    CalibrationDataAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.default_calibration_data_adapter import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEQ_LENGTH,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_loader_adapter import (
    ModelLoaderAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "model_preparation"


class ExecuTorchModelPreparationStrategy(ModelPreparationStrategy):
    """ExecuTorch-based model preparation using HuggingFace transformers.

    Delegates to injectable adapters for all external API calls, enabling
    dependency injection for testability:

    * ``ModelLoaderAdapter`` acquires the model, its tokenizer, and the example
      inputs describing the model's ``torch.export`` signature.
    * ``CalibrationDataAdapter`` produces the calibration corpus. Datasets are a
      cross-stage concern -- the same corpus feeds PTQ calibration and on-device
      evaluation -- so they live in ``genai_pipeline.datasets`` rather than being
      a model-loading responsibility.

    The model preparation flow:
    1. Validate the input configuration
    2. Load the model
    3. Load the tokenizer
    4. Build the export example inputs from the model
    5. Generate calibration data
    6. Optionally export the tokenizer for runtime use
    7. Extract the chat template

    Args:
        model_loader_adapter: Injectable adapter for model and tokenizer
            loading. Defaults to ``DefaultModelLoaderAdapter`` if not provided.
        calibration_data_adapter: Injectable adapter for calibration data
            generation. Defaults to ``DefaultCalibrationDataAdapter`` if not
            provided.
    """

    def __init__(
        self,
        model_loader_adapter: Optional[ModelLoaderAdapter] = None,
        calibration_data_adapter: Optional[CalibrationDataAdapter] = None,
    ) -> None:
        if model_loader_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.default_model_loader_adapter import (
                DefaultModelLoaderAdapter,
            )

            model_loader_adapter = DefaultModelLoaderAdapter()
        self._adapter = model_loader_adapter

        if calibration_data_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.datasets.default_calibration_data_adapter import (
                DefaultCalibrationDataAdapter,
            )

            calibration_data_adapter = DefaultCalibrationDataAdapter()
        self._calibration_adapter = calibration_data_adapter

    @property
    def adapter(self) -> ModelLoaderAdapter:
        """The model loader adapter used by this strategy."""
        return self._adapter

    @property
    def calibration_data_adapter(self) -> CalibrationDataAdapter:
        """The calibration data adapter used by this strategy."""
        return self._calibration_adapter

    def invoke(
        self,
        context: PipelineContext,
        input_config: ModelPreparationInputConfig,
    ) -> ModelPreparationOutputConfig:
        """Prepare the model, tokenizer, and calibration data.

        Args:
            context: The pipeline context with global settings.
            input_config: The model preparation input configuration.
                Supported keys in ``input_config.extra_options``:
                    - ``model_options``: Dict passed to ``load_model(extra_options=...)``.
                    - ``tokenizer_options``: Dict passed to ``load_tokenizer(extra_options=...)``.
                    - ``example_input_options``: Dict passed to
                      ``get_example_inputs(extra_options=...)``.
                    - ``num_calibration_samples``: Number of calibration samples
                      (default: ``DEFAULT_NUM_SAMPLES``).
                    - ``calibration_seq_length``: Sequence length per sample
                      (default: ``DEFAULT_SEQ_LENGTH``).
                    - ``calibration_options``: Dict passed to ``generate_calibration_data(extra_options=...)``.
                    - ``export_tokenizer``: If True, export tokenizer for runtime (default: False).
                    - ``tokenizer_export_options``: Dict passed to ``export_tokenizer(extra_options=...)``.
                    - ``chat_template``: Explicit chat template, used only when the
                      tokenizer does not carry one.

        Returns:
            ModelPreparationOutputConfig with model, tokenizer, and calibration data.

        Raises:
            StageError: If model_name is missing or any loading step fails.
        """
        logger.info(
            "Starting model preparation for '%s' on SoC=%s",
            input_config.model_name,
            input_config.soc_model,
        )

        self._validate_input(input_config)

        try:
            extra = dict(input_config.extra_options)

            # Step 1: Load model
            logger.debug("Loading model")
            model_module = self._adapter.load_model(
                model_name=input_config.model_name,
                extra_options=extra.get("model_options"),
            )

            # Step 2: Load tokenizer
            logger.debug("Loading tokenizer")
            tokenizer = self._adapter.load_tokenizer(
                model_name=input_config.model_name,
                extra_options=extra.get("tokenizer_options"),
            )

            # Step 3: Build the export example inputs from the model itself.
            # These are deliberately *not* taken from the calibration dataset:
            # they define the exported graph's positional signature (including
            # zero-initialized KV caches, which no dataset sample carries) and
            # the dataset's own attention-mask schema is derived from them.
            logger.debug("Building example inputs for export")
            example_inputs = self._adapter.get_example_inputs(
                model=model_module,
                extra_options=extra.get("example_input_options"),
            )

            # Step 4: Generate calibration data via the cross-stage dataset adapter
            logger.debug("Generating calibration data")
            num_samples = extra.get("num_calibration_samples", DEFAULT_NUM_SAMPLES)
            seq_length = extra.get("calibration_seq_length", DEFAULT_SEQ_LENGTH)
            calibration_data = self._calibration_adapter.generate_calibration_data(
                tokenizer=tokenizer,
                num_samples=num_samples,
                seq_length=seq_length,
                extra_options=extra.get("calibration_options"),
            )

            # Step 5: Optionally export tokenizer for runtime
            runtime_tokenizer_path = None
            if extra.get("export_tokenizer", False):
                logger.debug("Exporting tokenizer for runtime use")
                output_dir = Path(context.artifact_dir) / "tokenizer"
                runtime_tokenizer_path = self._adapter.export_tokenizer(
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    extra_options=extra.get("tokenizer_export_options"),
                )

            # Step 6: Extract chat_template from tokenizer (for instruct models).
            # The tokenizer wins over extra_options: a template shipped with the
            # model is authoritative, and extra_options is only a fallback for
            # models that carry none.
            chat_template = None
            if getattr(tokenizer, "chat_template", None):
                chat_template = tokenizer.chat_template
                logger.debug("Chat template extracted from tokenizer")
            elif extra.get("chat_template"):
                chat_template = extra["chat_template"]
                logger.debug("Chat template provided via extra_options")

            logger.info("Model preparation completed successfully")

            return ModelPreparationOutputConfig(
                model_module=model_module,
                tokenizer=tokenizer,
                example_inputs=example_inputs,
                calibration_data=calibration_data,
                runtime_tokenizer_path=runtime_tokenizer_path,
                chat_template=chat_template,
            )

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Model preparation failed",
                original_exception=e,
            ) from e

    def _validate_input(self, input_config: ModelPreparationInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The model preparation input configuration.

        Raises:
            StageError: If required fields are missing.
        """
        if not input_config.model_name:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="model_name is required for model preparation",
            )
        if not input_config.soc_model:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="soc_model is required for model preparation",
            )
