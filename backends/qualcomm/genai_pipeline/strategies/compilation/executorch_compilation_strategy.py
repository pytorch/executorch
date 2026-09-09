# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilerAdapter,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = "compilation"

# Optional lowering knobs forwarded from ``context.extra_options``. Required
# lowering inputs are explicit parameters on ``CompilerAdapter.compile_model``
# and deliberately not accepted here.
_COMPILE_EXTRA_KEYS = (
    "generate_etrecord",
    "skip_node_id_set",
    "skip_node_op_set",
    "skip_mutable_buffer",
    "convert_linear_to_conv2d",
    "executorch_backend_config",
)


class ExecuTorchCompilationStrategy(CompilationStrategy):
    """ExecuTorch-based compilation using QNN compiler backend.

    Delegates to a ``CompilerAdapter`` for all external API calls,
    enabling dependency injection for testability.

    The compilation flow:
    1. Validate the input configuration
    2. Delegate to the compiler adapter (wrapping
       ``to_edge_transform_and_lower_to_qnn``)
    3. Return artifact paths and optional ETRecord

    .. note::
        **Single-graph only; multi-graph is a follow-up.** This strategy makes
        exactly **one** ``compile_model`` call, mirroring the single-graph
        contract of the adapter it delegates to. Models exported as several
        graphs from the same weights (the hybrid AR-N prefill / AR-1 decode
        pair, plus an optional token-embedding graph) need a loop over the
        *deployed* graphs, which is why it is deliberately not attempted here:

        * only deployed graphs reach compilation: the full-auto-regressive
          calibration graph a hybrid decoder also builds exists purely to
          source quantization encodings, is reconciled into the deployed graphs
          by the quantization stage and released there, so the loop is over
          prefill and decode -- not over every graph the model preparation
          stage produced;
        * each deployed graph carries its own module, example inputs and
          compile specs, so the lowering call takes graph-name-keyed dicts
          rather than single values. ExecuTorch's
          ``to_edge_transform_and_lower_to_qnn`` already accepts those dicts,
          so this needs *using*, not building;
        * grouping the graphs into one multi-method ``.pte`` (weight sharing) is
          a property of that single lowering call, so it cannot be recovered by
          calling this single-graph path repeatedly.

        So the eventual interface is a per-graph map of module to example inputs
        and compile specs, which is additive to ``CompilationInputConfig``, so
        deferring costs nothing structurally. Per the layering used throughout
        this package, the fan-out belongs in this *strategy* -- adapters stay
        thin 1:1 wrappers over one graph. ``CompilationOutputConfig.artifact_paths``
        likewise stays a ``List`` here and becomes graph-name-keyed with the same
        change.

    Args:
        compiler_adapter: Injectable adapter for compilation operations.
            Defaults to ``DefaultCompilerAdapter`` if not provided.
    """

    def __init__(
        self,
        compiler_adapter: Optional[CompilerAdapter] = None,
    ) -> None:
        if compiler_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
                DefaultCompilerAdapter,
            )

            compiler_adapter = DefaultCompilerAdapter()
        self._adapter = compiler_adapter

    @property
    def adapter(self) -> CompilerAdapter:
        """The compiler adapter used by this strategy."""
        return self._adapter

    def invoke(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
    ) -> CompilationOutputConfig:
        """Compile the model to on-device .pte artifacts.

        Args:
            context: The pipeline context with global settings.
            input_config: The compilation input configuration.

        Returns:
            CompilationOutputConfig with artifact paths and optional ETRecord.

        Raises:
            StageError: If the model or example inputs are missing, or if
                compilation fails.
        """
        self._validate_input(input_config)

        logger.info(
            "Starting compilation for model '%s' on SoC=%s, backend=%s",
            context.model_name,
            input_config.soc_model,
            input_config.backend_type,
        )

        try:
            artifact_dir = Path(input_config.artifact_dir)
            file_name = context.model_name

            # Pass compilation-relevant options from context.extra_options.
            # Only forward keys that the compiler adapter may need, avoiding
            # leaking unrelated pipeline options.
            compile_extra = {
                k: v
                for k, v in context.extra_options.items()
                if k in _COMPILE_EXTRA_KEYS
            }

            # ``example_inputs`` is passed explicitly rather than through
            # ``extra_options``: it comes from the model (see
            # ``CompilationInputConfig.example_inputs``) and ``torch.export``
            # cannot run without it, so it is part of the adapter's signature.
            result = self._adapter.compile_model(
                model=input_config.model,
                example_inputs=input_config.example_inputs,
                compile_specs=input_config.compile_specs,
                artifact_dir=artifact_dir,
                file_name=file_name,
                soc_model=input_config.soc_model,
                backend_type=input_config.backend_type,
                extra_options=compile_extra if compile_extra else None,
            )

            logger.info(
                "Compilation completed: %d artifact(s) produced",
                len(result.artifact_paths),
            )

            return CompilationOutputConfig(
                artifact_paths=result.artifact_paths,
                etrecord=result.etrecord,
            )

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Compilation failed",
                original_exception=e,
            ) from e

    def _validate_input(self, input_config: CompilationInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The compilation input configuration.

        Raises:
            StageError: If required fields are missing.
        """
        if input_config.model is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="model is required for compilation",
            )
        # ``is None`` rather than a truthiness test: an empty tuple is a valid
        # export signature for a model taking no positional inputs, and only the
        # field's absence means the previous stage produced nothing. Same
        # reasoning as ``calibration_data`` in the quantization strategy.
        if input_config.example_inputs is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    "example_inputs is required for compilation; it is produced "
                    "from the model by ModelLoaderAdapter.get_example_inputs"
                ),
            )
        if input_config.soc_model is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="soc_model is required for compilation",
            )
        if input_config.backend_type is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="backend_type is required for compilation",
            )
