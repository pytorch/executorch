# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Optional

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantizer_adapter import (
    QuantizerAdapter,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "quantization"


class ExecuTorchQuantizationStrategy(QuantizationStrategy):
    """ExecuTorch-based quantization using QNN quantizer annotator rules.

    Delegates to a ``QuantizerAdapter`` for all external API calls,
    enabling dependency injection for testability.

    The quantization flow follows the PT2E pattern:
    1. Export the model via ``torch.export`` using ``input_config.example_inputs``
    2. Create a QNN quantizer with appropriate backend rules
    3. Prepare the model (insert observers)
    4. Calibrate with provided dataset
    5. Convert to quantized model

    .. note::
        **Single-graph only; multi-graph is a tracked follow-up.** This sequence
        runs one ``prepare_pt2e`` / calibrate / ``convert_pt2e`` pass over one
        module. Models exported as several graphs from the same weights (the
        hybrid AR-N prefill / AR-1 decode pair) need a different, *asymmetric*
        orchestration, which is why it is deliberately not attempted here:

        * every graph runs the full ``prepare_pt2e`` -> run -> ``convert_pt2e``
          sequence, but the data fed in between differs per graph: the
          calibration-only graph (full AR sequence with KV cache, never
          deployed) receives the real dataset, while the deployed prefill and
          decode graphs receive their own ``example_inputs`` -- one pass is
          still required there, or ``convert_pt2e`` fails on uninitialized observers;
        * the scales/zero-points collected on the calibration graph are then
          propagated to prefill and decode by an encoding-reconciliation step.

        So the eventual interface is a per-graph map of module to data source, plus a
        reconciliation hook, not a bare Dict[str, nn.Module].
        Landing that requires graph-map fields on the quantization configs, which are
        additive to this dataclass, so deferring costs nothing structurally.
        Per the layering used throughout this package, the fan-out belongs in
        this *strategy* -- adapters stay thin 1:1 wrappers over one graph.

    Args:
        quantizer_adapter: Injectable adapter for quantization operations.
            Defaults to ``DefaultQuantizerAdapter`` if not provided.
    """

    def __init__(
        self,
        quantizer_adapter: Optional[QuantizerAdapter] = None,
    ) -> None:
        if quantizer_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.default_quantizer_adapter import (
                DefaultQuantizerAdapter,
            )

            quantizer_adapter = DefaultQuantizerAdapter()
        self._adapter = quantizer_adapter

    @property
    def adapter(self) -> QuantizerAdapter:
        """The quantizer adapter used by this strategy."""
        return self._adapter

    def invoke(
        self,
        context: PipelineContext,
        input_config: QuantizationInputConfig,
    ) -> QuantizationOutputConfig:
        """Quantize the model using ExecuTorch/QNN quantization.

        Executes the full PT2E quantization pipeline:
        export → make_quantizer → prepare_pt2e → calibrate → convert_pt2e.

        Args:
            context: The pipeline context with global settings.
            input_config: The quantization input configuration.

        Returns:
            QuantizationOutputConfig with the quantized model.

        Raises:
            StageError: If the model, example inputs or calibration data is
                missing, or if any quantization step fails.
        """
        logger.info(
            "Starting quantization for model '%s' on SoC=%s, backend=%s",
            context.model_name,
            input_config.soc_model,
            input_config.backend_type,
        )

        self._validate_input(input_config)

        if input_config.training_data is not None:
            # QAT is not yet wired up: ``training_data`` is carried on the config
            # (mirroring ``qat_training_data`` in ``build_executorch_binary``) so
            # the contract is stable, but this strategy only implements PTQ.
            logger.warning(
                "training_data was provided but quantization-aware training is "
                "not implemented by this strategy; proceeding with PTQ."
            )

        try:
            # Step 1: Export model. The example inputs describe the model's own
            # export signature (see ``QuantizationInputConfig.example_inputs``);
            # they are never drawn from ``calibration_data``, which is left
            # untouched so that even a single-use generator reaches ``calibrate``
            # with every sample intact.
            logger.debug("Exporting model")
            exported_model = self._adapter.export_model(
                input_config.model_module,
                input_config.example_inputs,
            )

            # Step 2: Create quantizer
            logger.debug("Creating quantizer")
            quant_kwargs = dict(input_config.extra_options)
            quant_dtype = quant_kwargs.pop("quant_dtype", None)
            quant_recipe = quant_kwargs.pop("quant_recipe", None) or getattr(
                input_config, "quant_recipe", None
            )

            # Build make_quantizer arguments — only pass quant_dtype if
            # explicitly provided, so the default owned by
            # ``export_utils.make_quantizer`` (use_8a8w) applies otherwise
            # instead of being shadowed by a value chosen here.
            # ``quant_recipe`` is not an argument of that function: the adapter
            # consumes it and applies it to the constructed quantizer via
            # ``QnnQuantizer.set_recipe``.
            make_quantizer_kwargs = {
                "backend": input_config.backend_type,
                "soc_model": input_config.soc_model,
                **quant_kwargs,
            }
            if quant_dtype is not None:
                make_quantizer_kwargs["quant_dtype"] = quant_dtype
            if quant_recipe is not None:
                make_quantizer_kwargs["quant_recipe"] = quant_recipe

            quantizer = self._adapter.make_quantizer(**make_quantizer_kwargs)

            # Step 3: Prepare (insert observers)
            logger.debug("Preparing model for quantization")
            annotated_model = self._adapter.prepare_pt2e(exported_model, quantizer)

            # Step 4: Calibrate
            logger.debug("Running calibration")
            calibrated_model = self._adapter.calibrate(
                annotated_model, input_config.calibration_data
            )

            # Step 5: Convert to quantized model
            logger.debug("Converting to quantized model")
            quantized_model = self._adapter.convert_pt2e(calibrated_model)

            logger.info("Quantization completed successfully")
            return QuantizationOutputConfig(quantized_model=quantized_model)

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Quantization failed",
                original_exception=e,
            ) from e

    def _validate_input(self, input_config: QuantizationInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The quantization input configuration.

        Raises:
            StageError: If required fields are missing.
        """
        if input_config.model_module is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="model_module is required for quantization",
            )
        if input_config.example_inputs is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    "example_inputs is required for quantization; it is produced "
                    "from the model by ModelLoaderAdapter.get_example_inputs"
                ),
            )
        # ``is None`` rather than a truthiness test: ``calibration_data`` may be a
        # generator or ``DataLoader``, and ``not <iterator>`` would consume the
        # first sample without reliably detecting emptiness.
        if input_config.calibration_data is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="calibration_data is required for quantization",
            )
        if input_config.soc_model is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="soc_model is required for quantization",
            )
        if input_config.backend_type is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="backend_type is required for quantization",
            )
