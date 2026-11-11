# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.component_spec_policy import (
    build_component_compile_specs,
    build_executorch_config,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.pass_policy import (
    build_component_pass_config,
    resolve_num_sharding,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.compilation_output_config import (
    CompilationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compilation_strategy import (
    CompilationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilerAdapter,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle

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
)


class ExecuTorchCompilationStrategy(CompilationStrategy):
    """ExecuTorch-based compilation using QNN compiler backend.

    Delegates to a ``CompilerAdapter`` for all external API calls, enabling
    dependency injection for testability.

    Takes one of two paths, depending on how the input names what to compile:

    * ``input_config.graphs`` -- one ``compile_graph_group`` call per component,
      each producing one ``.pte``. A component's graphs go into a single call so
      that they share weights, which is why fanning out over *components* is
      this strategy's job while grouping *graphs* is the adapter's.
    * ``input_config.model`` -- a single ``compile_model`` call, for a model with
      one graph.

    Deriving the compile specs and the per-graph pass configuration also belongs
    here rather than upstream: both depend on which component a graph belongs to
    and on how the graphs are grouped, neither of which the quantization stage
    knows. Deriving them together also keeps them consistent -- the shard count
    that requests multiple contexts is the one that inserts the splitting pass.

    Args:
        compiler_adapter: Injectable adapter for compilation operations.
            Defaults to ``DefaultCompilerAdapter``. An LLM flow injects
            ``LLMCompilerAdapter`` instead, which is what keeps the decoder's
            model-specific lowering settings out of this strategy.
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
            StageError: If the input names nothing to compile, or if compilation
                fails.
        """
        self._validate_input(input_config)

        logger.info(
            "Starting compilation for model '%s' on SoC=%s, backend=%s",
            context.model_name,
            input_config.soc_model,
            input_config.backend_type,
        )

        try:
            graphs = input_config.graphs or self._unquantized_graphs(input_config)
            if graphs:
                self._validate_component_graphs(graphs)
                output = self._compile_components(context, input_config, graphs)
            else:
                output = self._compile_single_graph(context, input_config)

            logger.info(
                "Compilation completed: %d artifact(s) produced (%s)",
                len(output.artifact_paths or {}),
                ", ".join(sorted(output.artifact_paths or {})),
            )
            return output

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Compilation failed",
                original_exception=e,
            ) from e

    def _compile_single_graph(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
    ) -> CompilationOutputConfig:
        """Compile a model that has exactly one graph.

        Args:
            context: The pipeline context with global settings.
            input_config: The compilation input configuration.

        Returns:
            CompilationOutputConfig with the one artifact path.
        """
        # ``example_inputs`` is passed explicitly rather than through
        # ``extra_options``: it comes from the model (see
        # ``CompilationInputConfig.example_inputs``) and ``torch.export``
        # cannot run without it, so it is part of the adapter's signature.
        result = self._adapter.compile_model(
            model=input_config.model,
            example_inputs=input_config.example_inputs,
            compile_specs=input_config.compile_specs,
            artifact_dir=Path(input_config.artifact_dir),
            file_name=context.model_name,
            soc_model=input_config.soc_model,
            backend_type=input_config.backend_type,
            executorch_config=build_executorch_config(ARTIFACT_TEXT_DECODER),
            extra_options=self._compile_extra(context),
        )

        return CompilationOutputConfig(
            artifact_paths=result.artifact_paths,
            etrecord=result.etrecord,
        )

    @staticmethod
    def _unquantized_graphs(
        input_config: CompilationInputConfig,
    ) -> Optional[Dict[str, Dict[str, "GraphBundle"]]]:
        """Bundle a component-keyed model whose quantization was skipped.

        Args:
            input_config: The compilation input configuration.

        Returns:
            Unquantized bundles, or ``None`` for a single-graph model.
        """
        if not isinstance(input_config.model, dict):
            return None

        from executorch.backends.qualcomm.genai_pipeline.unquantized_bundles import (
            build_unquantized_bundles,
        )

        logger.info("Quantization was skipped; compiling unquantized graphs")
        return build_unquantized_bundles(
            model_module=input_config.model,
            example_inputs=input_config.example_inputs,
            meta=input_config.meta,
        )

    @staticmethod
    def _component_quantized(
        artifact_key: str,
        graphs: Dict[str, "GraphBundle"],
    ) -> bool:
        """Whether one component's graphs were quantized.

        Args:
            artifact_key: The component being compiled.
            graphs: The component's graphs, keyed by graph name.

        Returns:
            The component's shared ``quantized`` flag.

        Raises:
            StageError: If the component mixes quantized and unquantized graphs,
                since one ``.pte`` gets one precision.
        """
        flags = {bundle.quantized for bundle in graphs.values()}
        if len(flags) > 1:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    f"component '{artifact_key}' mixes quantized and unquantized "
                    "graphs"
                ),
            )
        return flags.pop()

    def _compile_components(
        self,
        context: PipelineContext,
        input_config: CompilationInputConfig,
        all_graphs: Dict[str, Dict[str, "GraphBundle"]],
    ) -> CompilationOutputConfig:
        """Compile each component into its own ``.pte``.

        Args:
            context: The pipeline context with global settings.
            input_config: The compilation input configuration.
            all_graphs: The graphs to compile, keyed by artifact key then graph
                name.

        Returns:
            CompilationOutputConfig holding one path per compiled component.
            Components the model does not have are simply absent.
        """
        control_args = ControlArgs.from_pipeline_context(context)
        builder = self._build_spec_builder(control_args, input_config)
        extra_options = self._compile_extra(context)

        artifact_paths: Dict[str, Path] = {}
        etrecord = None
        etrecord_source = None

        # The token embedding's quantized IO is tagged against the *decoder's*
        # shapes, as in the legacy flow, so the decoder's graphs have to be
        # reachable while another component is being configured.
        decoder_graphs = all_graphs.get(ARTIFACT_TEXT_DECODER)

        for artifact_key, graphs in all_graphs.items():
            # One count for both specs and passes, so multiple contexts are
            # requested only for a graph that is really split.
            num_sharding = resolve_num_sharding(
                artifact_key,
                graphs,
                self._component_num_sharding(input_config, artifact_key),
            )
            compile_specs = build_component_compile_specs(
                builder=builder,
                artifact_key=artifact_key,
                graph_names=list(graphs),
                control_args=control_args,
                num_sharding=num_sharding,
                quantized=self._component_quantized(artifact_key, graphs),
            )
            dep_table, passes_job = build_component_pass_config(
                artifact_key=artifact_key,
                graphs=graphs,
                control_args=control_args,
                num_sharding=num_sharding,
                decoder_graphs=decoder_graphs,
            )

            logger.debug(
                "Compiling component '%s' from graph(s) %s",
                artifact_key,
                sorted(graphs),
            )
            result = self._adapter.compile_graph_group(
                graphs=graphs,
                compile_specs=compile_specs,
                artifact_dir=Path(input_config.artifact_dir),
                file_name=f"{context.model_name}_{artifact_key}",
                soc_model=input_config.soc_model,
                backend_type=input_config.backend_type,
                executorch_config=build_executorch_config(artifact_key),
                dep_table=dep_table,
                passes_job=passes_job,
                artifact_key=artifact_key,
                extra_options=extra_options,
            )

            artifact_paths.update(result.artifact_paths)
            if etrecord is None and result.etrecord is not None:
                etrecord, etrecord_source = result.etrecord, artifact_key

        if etrecord_source is not None:
            # CompilationOutputConfig holds one ETRecord; the first one wins.
            logger.info("Using the ETRecord of component '%s'", etrecord_source)

        return CompilationOutputConfig(
            artifact_paths=artifact_paths,
            etrecord=etrecord,
        )

    @staticmethod
    def _component_num_sharding(
        input_config: CompilationInputConfig,
        artifact_key: str,
    ) -> int:
        """The shard count requested for one component.

        ``num_sharding`` is per-component, since an encoder's shard count comes
        from its own config rather than the decoder's
        (``GraniteSpeechEncoder.num_sharding``). A plain ``int`` is still
        accepted and taken to describe the decoder, which is what a single-shard
        text-only flow passes.

        Args:
            input_config: The compilation input configuration.
            artifact_key: The component being compiled.

        Returns:
            The requested shard count, 1 when it is not described.
        """
        num_sharding = input_config.num_sharding
        if isinstance(num_sharding, dict):
            return int(num_sharding.get(artifact_key, 1))

        # A scalar describes the decoder only.
        if artifact_key == ARTIFACT_TEXT_DECODER:
            return int(num_sharding)

        return 1

    @staticmethod
    def _build_spec_builder(
        control_args: ControlArgs,
        input_config: CompilationInputConfig,
    ) -> Any:
        """Construct the compile spec builder for this run's target.

        One builder describes one target, which is all a single-backend model
        needs. Splitting components across backends would need one builder per
        backend, so this is the place that would grow a mapping.

        Args:
            control_args: The flow's control arguments.
            input_config: The compilation input configuration, whose
                ``soc_model`` is authoritative.

        Returns:
            A ``QnnCompileSpecBuilder`` for the target.
        """
        from executorch.backends.qualcomm.genai_pipeline.compilation.compile_spec_builder import (
            QnnCompileSpecBuilder,
        )

        return QnnCompileSpecBuilder(
            soc_model=input_config.soc_model,
            backend=control_args.backend,
            enable_x86_64=control_args.enable_x86_64,
        )

    @staticmethod
    def _compile_extra(context: PipelineContext) -> Optional[Dict[str, Any]]:
        """Collect the lowering knobs this stage forwards from the context.

        Only the keys the compiler adapter may need are forwarded, so unrelated
        pipeline options do not leak into lowering.

        Args:
            context: The pipeline context with global settings.

        Returns:
            The forwarded options, or ``None`` if there are none.
        """
        extra = {
            k: v for k, v in context.extra_options.items() if k in _COMPILE_EXTRA_KEYS
        }
        return extra or None

    def _validate_input(self, input_config: CompilationInputConfig) -> None:
        """Validate required fields in the input configuration.

        Args:
            input_config: The compilation input configuration.

        Raises:
            StageError: If required fields are missing.
        """
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

        if input_config.graphs:
            self._validate_component_graphs(input_config.graphs)
            return

        if input_config.model is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    "compilation needs either graphs, produced by the "
                    "quantization stage, or a single model"
                ),
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

    @staticmethod
    def _validate_component_graphs(
        graphs: Dict[str, Dict[str, "GraphBundle"]],
    ) -> None:
        """Check the component map is well formed before any lowering starts.

        Args:
            graphs: The graphs to compile, keyed by artifact key then graph name.

        Raises:
            StageError: If a component names an unknown artifact key, or has no
                graphs -- the latter would otherwise produce an empty ``.pte``.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ALL_ARTIFACT_KEYS,
        )

        unknown = sorted(set(graphs) - ALL_ARTIFACT_KEYS)
        if unknown:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    f"graphs names unknown artifact key(s) {unknown}; valid "
                    f"keys: {sorted(ALL_ARTIFACT_KEYS)}"
                ),
            )

        empty = sorted(key for key, component in graphs.items() if not component)
        if empty:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=f"component(s) {empty} have no graphs to compile",
            )
