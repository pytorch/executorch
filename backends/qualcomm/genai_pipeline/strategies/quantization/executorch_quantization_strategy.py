# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import logging
from enum import auto, Enum
from typing import Any, Optional, TYPE_CHECKING

import torch
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.exceptions import StageError
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    GRAPH_KV_FORWARD,
    GRAPH_PREFILL_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantizer_adapter import (
    QuantizerAdapter,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.datasets.calibration.calibration_data_adapter import (
        CalibrationDataAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.datasets.evaluation.evaluation_data_adapter import (
        EvaluationDataAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.datasets.training.training_data_adapter import (
        TrainingDataAdapter,
    )

logger = logging.getLogger(__name__)

_STAGE_NAME = "quantization"


class _GraphRole(Enum):
    # Uses real calibration data but is never lowered for deployment.
    QUANTIZE = auto()
    # Initializes encodings, receives overrides from a quantize graph, then lowers.
    DEPLOY = auto()
    # Uses calibration data and lowers as the same graph; no encoding initialization.
    SHARED = auto()


_GRAPH_ROLES = {
    # The quantize graph supplies encodings for the decoder's deployed variants.
    ARTIFACT_TEXT_DECODER: {
        GRAPH_FORWARD: _GraphRole.QUANTIZE,
        **dict.fromkeys(DECODER_GRAPH_NAMES, _GraphRole.DEPLOY),
    },
    # The quantize graph supplies encodings for deployed token-embedding variants.
    ARTIFACT_TOK_EMBEDDING: {
        GRAPH_FORWARD: _GraphRole.QUANTIZE,
        **dict.fromkeys(TOK_EMBEDDING_GRAPH_NAMES, _GraphRole.DEPLOY),
    },
    # Audio-encoder graph is quantized and lowered for deployment.
    ARTIFACT_AUDIO_ENCODER: {GRAPH_FORWARD: _GraphRole.SHARED},
    # Text-encoder graph is quantized and lowered for deployment.
    ARTIFACT_TEXT_ENCODER: {GRAPH_FORWARD: _GraphRole.SHARED},
    # Vision-encoder graph is quantized and lowered for deployment.
    ARTIFACT_VISION_ENCODER: {GRAPH_FORWARD: _GraphRole.SHARED},
}


class ExecuTorchQuantizationStrategy(QuantizationStrategy):
    """ExecuTorch-based quantization using QNN quantizer annotator rules.

    Delegates single-graph PT2E operations to ``QuantizerAdapter`` and owns the
    component/graph routing. Each graph has one of three roles:

    - ``QUANTIZE``: Runs on real calibration data to produce encodings, then is
      removed without being deployed.
    - ``DEPLOY``: Runs once to initialize observers, receives encodings from its
      component's quantize graph, then is lowered for deployment.
    - ``SHARED``: Runs on calibration data and is lowered as the same graph.

    The quantization flow mirrors :meth:`invoke`:

    1. Create QNN quantizers and recipe instances for every component graph.
    2. Export graph variants and prepare them with observers.
    3. Initialize observers for separate deploy graphs.
    4. Assemble real-data inputs for native PTQ calibration.
    5. Quantize ``QUANTIZE`` and ``SHARED`` graphs.
    6. Convert prepared graphs to QDQ modules.
    7. Save decoder QDQ output and override quantize-graph encodings to deploy
       variants, then release quantize-only graphs.

    Args:
        quantizer_adapter: Injectable adapter for single-graph quantization
            operations. Defaults to ``DefaultQuantizerAdapter`` if not provided.
        calibration_data_adapter: Injectable purpose adapter that assembles
            calibration data. Defaults to ``DefaultCalibrationDataAdapter`` (random
            fallback) when not provided.
        training_data_adapter: Injectable adapter for QAT training data. Retained
            for the QAT flow; the current native PTQ implementation does not
            consume it.
        evaluation_data_adapter: Injectable adapter for post-quantization
            evaluation data. Retained for evaluation integration; the current
            strategy does not run evaluation or consume it.
    """

    def __init__(
        self,
        quantizer_adapter: Optional[QuantizerAdapter] = None,
        calibration_data_adapter: Optional["CalibrationDataAdapter"] = None,
        training_data_adapter: Optional["TrainingDataAdapter"] = None,
        evaluation_data_adapter: Optional["EvaluationDataAdapter"] = None,
    ) -> None:
        if quantizer_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.default_quantizer_adapter import (
                DefaultQuantizerAdapter,
            )

            quantizer_adapter = DefaultQuantizerAdapter()
        self._adapter = quantizer_adapter
        if calibration_data_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.datasets.calibration.default_calibration_data_adapter import (
                DefaultCalibrationDataAdapter,
            )

            calibration_data_adapter = DefaultCalibrationDataAdapter()
        self._calibration_data_adapter = calibration_data_adapter
        if training_data_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.datasets.training.default_training_data_adapter import (
                DefaultTrainingDataAdapter,
            )

            training_data_adapter = DefaultTrainingDataAdapter()
        self._training_data_adapter = training_data_adapter
        if evaluation_data_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.datasets.evaluation.default_evaluation_data_adapter import (
                DefaultEvaluationDataAdapter,
            )

            evaluation_data_adapter = DefaultEvaluationDataAdapter()
        self._evaluation_data_adapter = evaluation_data_adapter

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

        The native PTQ flow creates graph quantizers, exports and prepares every
        graph, initializes deploy-graph observers, quantizes graphs that consume
        calibration data, converts to QDQ, and propagates encodings to deploy
        variants.

        Args:
            context: The pipeline context with global settings.
            input_config: The quantization input configuration.

        Returns:
            Component- and graph-keyed bundles for deployment graphs. Shared
            graphs remain in the output; quantize-only graphs are removed after
            encoding propagation.

        Raises:
            StageError: If required input is missing or any quantization step
                fails.
        """
        logger.info(
            "Starting quantization for model '%s' on SoC=%s, backend=%s",
            context.model_name,
            input_config.soc_model,
            input_config.backend_type,
        )

        self._validate_input(input_config)

        try:
            extra = input_config.extra_options or {}
            quant_options = dict(extra.get("quantize_options") or {})

            # Step 1: Create QNN quantizers and per-graph recipe instances.
            quantizers, quant_recipes = self._make_quantizer(
                input_config,
                quant_options,
            )

            # Step 2: Export graph variants and prepare them with observers.
            prepared_modules = self._export_and_prepare(input_config, quantizers)

            # Step 3: Initialize deploy-graph observers for QDQ conversion.
            self._initialize_encodings(prepared_modules, input_config)

            # Step 4: Assemble real-data inputs for native PTQ calibration.
            calibration_data = self._calibration_data_adapter.generate_calibration_data(
                tokenizer=input_config.tokenizer,
                example_inputs=self._flatten_calibration_example_inputs(
                    input_config.example_inputs
                ),
            )

            # Step 5: Quantize graphs that consume calibration data.
            self._calibrate(prepared_modules, input_config, calibration_data)

            # Step 6: Convert prepared graphs to QDQ modules.
            converted_modules = self._convert_pt2e(prepared_modules)

            # Step 7: Override quantize-graph encodings to deploy variants.
            self._override_encodings(converted_modules, input_config, context)

            logger.info("Quantization completed successfully")

            # Build deployment graph bundles.
            graphs = {}
            for component, converted_graphs in converted_modules.items():
                graphs[component] = {
                    graph_name: GraphBundle(
                        module=module,
                        inputs=input_config.example_inputs[component][graph_name],
                        meta=input_config.meta.get(component, {}).get(graph_name, {}),
                        quant_io_dtypes=self._get_quant_io_dtypes(
                            quant_recipes[component][graph_name]
                        ),
                    )
                    for graph_name, module in converted_graphs.items()
                }
            return QuantizationOutputConfig(graphs=graphs)

        except StageError:
            raise
        except Exception as e:
            raise StageError(
                stage_name=_STAGE_NAME,
                message="Quantization failed",
                original_exception=e,
            ) from e

    def _make_quantizer(
        self,
        input_config: QuantizationInputConfig,
        quant_options: dict,
    ) -> tuple[dict, dict]:
        """Build QNN quantizers and recipe instances for every component graph.

        Args:
            input_config: Model and backend configuration for quantization.
            quant_options: Component-keyed dtype and recipe configuration.

        Returns:
            A ``(quantizers, recipes)`` tuple, each keyed by component then
            graph name. Recipe classes are instantiated per graph; only the
            ``GRAPH_FORWARD`` recipe instance enables verbose output.
        """
        from executorch.backends.qualcomm.genai_pipeline.quant_utilities import (
            make_quantizer,
        )

        quant_dtype = quant_options.get("quant_dtype") or {}
        quant_recipe = quant_options.get("quant_recipe") or {}
        quantizers = {}
        recipe_instances = {}
        for component, component_example_inputs in input_config.example_inputs.items():
            quantizers[component] = {}
            recipe_instances[component] = {}
            recipe_class = quant_recipe.get(component)
            for graph_name in component_example_inputs:
                recipe = (
                    recipe_class(verbose=graph_name == GRAPH_FORWARD)
                    if isinstance(recipe_class, type)
                    else recipe_class
                )
                make_quantizer_kwargs = {
                    "backend": input_config.backend_type,
                    "soc_model": input_config.soc_model,
                    **(
                        {"quant_dtype": dtype}
                        if (dtype := quant_dtype.get(component)) is not None
                        else {}
                    ),
                    **({"quant_recipe": recipe} if recipe is not None else {}),
                }
                quantizers[component][graph_name] = make_quantizer(
                    **make_quantizer_kwargs
                )
                recipe_instances[component][graph_name] = recipe
                if recipe is not None:
                    logger.info(
                        "Created QNN quantizer for '%s' graph '%s' with quant "
                        "recipe %s",
                        component,
                        graph_name,
                        type(recipe).__name__,
                    )
                else:
                    logger.info(
                        "Quant recipe not set yet; created QNN quantizer for "
                        "'%s' graph '%s' with quant dtype %s",
                        component,
                        graph_name,
                        make_quantizer_kwargs.get("quant_dtype"),
                    )
        return quantizers, recipe_instances

    def _export_and_prepare(
        self,
        input_config: QuantizationInputConfig,
        quantizers: dict,
    ) -> dict:
        """Export every graph and prepare it for PT2E quantization.

        Text-decoder example inputs are flattened into the positional export
        signature before export. Other component inputs pass through unchanged.

        Args:
            input_config: Model modules, graph inputs, and graph metadata.
            quantizers: Component- and graph-keyed QNN quantizers.

        Returns:
            Component- and graph-keyed prepared PT2E modules.
        """
        prepared_modules = {}
        for component, component_example_inputs in input_config.example_inputs.items():
            prepared_modules[component] = {}
            for graph_name, graph_example_inputs in component_example_inputs.items():
                export_inputs = self._post_process_example_inputs(
                    component,
                    graph_name,
                    graph_example_inputs,
                    input_config.meta,
                )
                prepared_modules[component][graph_name] = self._adapter.prepare_pt2e(
                    self._adapter.export_model(
                        input_config.model_module[component], export_inputs
                    ),
                    quantizers[component][graph_name],
                )
        return prepared_modules

    @staticmethod
    def _post_process_example_inputs(
        component: str,
        graph_name: str,
        example_inputs: Any,
        meta: dict,
    ) -> Any:
        """Flatten text-decoder inputs into their positional export signature.

        Args:
            component: Artifact component key for the graph.
            graph_name: Graph key used to retrieve metadata.
            example_inputs: Model-preparation inputs in structured decoder form.
            meta: Component- and graph-keyed model metadata.

        Returns:
            Unchanged inputs for non-decoder components. For text decoder graphs,
            positional inputs with attention masks expanded and position IDs/KV
            caches included only when ``get_use_kv_cache`` is enabled.

        Raises:
            StageError: If a text decoder graph lacks ``get_use_kv_cache``
                metadata.
        """
        if component != ARTIFACT_TEXT_DECODER:
            return example_inputs

        if "get_use_kv_cache" not in (
            graph_meta := meta.get(component, {}).get(graph_name, {})
        ):
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    f"Component '{component}' graph '{graph_name}' is missing "
                    "get_use_kv_cache metadata"
                ),
            )

        use_kv_cache = graph_meta["get_use_kv_cache"]
        return (
            example_inputs[0],
            *example_inputs[1],
            *((example_inputs[2],) if use_kv_cache else []),
            *(example_inputs[3] if use_kv_cache else []),
            *(example_inputs[4] if use_kv_cache else []),
        )

    def _initialize_encodings(
        self,
        prepared_modules: dict,
        input_config: QuantizationInputConfig,
    ) -> None:
        """Initialize deployed graph observers with their export signatures.

        Deploy graphs do not consume the corpus calibration data because their
        encodings are later overridden from the component's quantize graph. They
        still require one forward pass so ``convert_pt2e`` can produce QDQ
        graphs.

        Args:
            prepared_modules: Component- and graph-keyed prepared PT2E modules.
            input_config: Graph inputs and metadata used for observer execution.
        """
        for component, prepared_graphs in prepared_modules.items():
            for graph_name, graph_module in prepared_graphs.items():
                if self._is_deploy_graph(component, graph_name):
                    self._adapter.init_encodings(
                        graph_module,
                        self._post_process_example_inputs(
                            component,
                            graph_name,
                            input_config.example_inputs[component][graph_name],
                            input_config.meta,
                        ),
                    )

    def _calibrate(
        self,
        prepared_modules: dict,
        input_config: QuantizationInputConfig,
        calibration_data: dict,
    ) -> None:
        """Run the quantization algorithm on graphs that consume real data.

        ``QUANTIZE`` graphs produce encodings for separate deploy variants;
        ``SHARED`` graphs produce their own encodings and are also deployed. The
        implementation currently supports only native PTQ calibration through
        the model-specific adapter.

        TODO: Support legacy quantization algorithms such as QAT and SeqMSE.

        Args:
            prepared_modules: Component- and graph-keyed prepared PT2E modules.
            input_config: Model-specific inference support for calibration.
            calibration_data: Component-keyed real-data calibration inputs.
        """
        quantization_graphs = {}
        for component, prepared_graphs in prepared_modules.items():
            for graph_name, graph_module in prepared_graphs.items():
                if not self._is_deploy_graph(component, graph_name):
                    quantization_graphs[component] = graph_module
        self._adapter.calibrate(
            quantization_graphs,
            calibration_data,
            inference=input_config.inference,
        )

    def _convert_pt2e(self, prepared_modules: dict) -> dict:
        """Convert prepared PT2E modules to QDQ modules and release inputs.

        Args:
            prepared_modules: Component- and graph-keyed prepared PT2E modules.

        Returns:
            Component- and graph-keyed converted QDQ modules.
        """
        converted_modules = {}
        for component, prepared_graphs in prepared_modules.items():
            converted_modules[component] = {
                graph_name: self._adapter.convert_pt2e(graph_module)
                for graph_name, graph_module in prepared_graphs.items()
            }
        prepared_modules.clear()
        gc.collect()
        return converted_modules

    def _override_encodings(
        self,
        converted_modules: dict,
        input_config: QuantizationInputConfig,
        context: PipelineContext,
    ) -> None:
        """Override quantize-graph encodings to separate deploy variants.

        For text decoder, saves the converted ``GRAPH_FORWARD`` QDQ module,
        then overrides decode and optional prefill graph encodings using cache
        layer metadata from the decode graph. For token embedding, overrides
        each configured deployment variant. Quantize-only graphs are removed
        after they have served as encoding sources.

        Args:
            converted_modules: Component- and graph-keyed converted QDQ modules.
            input_config: Decoder inputs and metadata needed for QDQ export and
                encoding propagation.
            context: Pipeline context providing the QDQ artifact directory.

        Raises:
            StageError: If a required quantize graph, deployed decoder graph, or
                decoder cache-layer metadata is missing.
        """
        from executorch.backends.qualcomm.genai_pipeline.quant_utilities import (
            encoding_override,
            save_logits_quant_attrs,
            save_output_kv_cache_quant_attrs,
            save_quantized_module,
        )

        if (decoder_graphs := converted_modules.get(ARTIFACT_TEXT_DECODER)) is not None:
            if (quantized_decoder := decoder_graphs.get(GRAPH_FORWARD)) is None:
                raise StageError(
                    stage_name=_STAGE_NAME,
                    message=(
                        f"Component '{ARTIFACT_TEXT_DECODER}' is missing its "
                        f"quantization graph '{GRAPH_FORWARD}'"
                    ),
                )
            try:
                # Saving Decoder QDQ Model EP
                save_quantized_module(
                    quantized_module=quantized_decoder,
                    example_inputs=self._post_process_example_inputs(
                        ARTIFACT_TEXT_DECODER,
                        GRAPH_FORWARD,
                        input_config.example_inputs[ARTIFACT_TEXT_DECODER][
                            GRAPH_FORWARD
                        ],
                        input_config.meta,
                    ),
                    artifact_dir=context.artifact_dir,
                )

                # Override decoder quant encodings
                if (decoder := decoder_graphs.get(GRAPH_KV_FORWARD)) is None:
                    raise StageError(
                        stage_name=_STAGE_NAME,
                        message=(
                            f"Component '{ARTIFACT_TEXT_DECODER}' is missing "
                            f"its deployed decoder graph '{GRAPH_KV_FORWARD}'"
                        ),
                    )
                decoder_meta = input_config.meta.get(ARTIFACT_TEXT_DECODER, {}).get(
                    GRAPH_KV_FORWARD, {}
                )
                if (
                    n_cache_layers := decoder_meta.get("get_n_self_layers")
                    or decoder_meta.get("get_n_layers")
                ) is None:
                    raise StageError(
                        stage_name=_STAGE_NAME,
                        message=(
                            f"Component '{ARTIFACT_TEXT_DECODER}' graph "
                            f"'{GRAPH_KV_FORWARD}' requires n_cache_layers metadata "
                            "for encoding override"
                        ),
                    )
                encoding_override(
                    quantized_model=quantized_decoder,
                    unquantized_model=decoder,
                    n_cache_layers=n_cache_layers,
                )
                save_logits_quant_attrs(decoder, decoder_meta)
                save_output_kv_cache_quant_attrs(decoder, decoder_meta)

                # Override prefill quant encodings
                if (prefill := decoder_graphs.get(GRAPH_PREFILL_FORWARD)) is not None:
                    prefill_meta = input_config.meta.get(ARTIFACT_TEXT_DECODER, {}).get(
                        GRAPH_PREFILL_FORWARD, {}
                    )
                    encoding_override(
                        quantized_model=quantized_decoder,
                        unquantized_model=prefill,
                        n_cache_layers=n_cache_layers,
                    )
                    save_logits_quant_attrs(prefill, prefill_meta)
                    save_output_kv_cache_quant_attrs(prefill, prefill_meta)
            finally:
                # The quantization graph only sources encodings; never deployed.
                decoder_graphs.pop(GRAPH_FORWARD, None)
                gc.collect()

        # Override token embedding quant encodings
        if (
            tok_embedding_graphs := converted_modules.get(ARTIFACT_TOK_EMBEDDING)
        ) is None:
            return
        if (quantized_tok_embedding := tok_embedding_graphs.get(GRAPH_FORWARD)) is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=(
                    f"Component '{ARTIFACT_TOK_EMBEDDING}' is missing its "
                    f"quantization graph '{GRAPH_FORWARD}'"
                ),
            )
        try:
            for graph_name in TOK_EMBEDDING_GRAPH_NAMES:
                if (tok_embedding := tok_embedding_graphs.get(graph_name)) is None:
                    continue
                encoding_override(
                    quantized_model=quantized_tok_embedding,
                    unquantized_model=tok_embedding,
                )
        finally:
            # The quantization graph only sources encodings; never deployed.
            tok_embedding_graphs.pop(GRAPH_FORWARD, None)
            gc.collect()

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
        if isinstance(input_config.backend_type, dict):
            raise StageError(
                stage_name=_STAGE_NAME,
                message="backend_type must be one shared QNN backend, not a map",
            )

    def _flatten_calibration_example_inputs(self, example_inputs: dict) -> dict:
        """Flatten ``{component: {graph_name: inputs}}`` to ``{component: inputs}``.

        Selects the first graph per component that participates in real-data
        quantization (``QUANTIZE`` or ``SHARED``) and drops the graph-name
        level. Separate deploy graphs are excluded because their KV-cache export
        signatures must not shape the collator's attention-mask template.

        Args:
            example_inputs: Component- and graph-keyed model example inputs.

        Returns:
            Component-keyed inputs for calibration data construction.
        """
        calibration_inputs = {}
        for component, graphs in example_inputs.items():
            for graph_name, inputs in graphs.items():
                if not self._is_deploy_graph(component, graph_name):
                    calibration_inputs[component] = inputs
                    break
        return calibration_inputs

    def _is_deploy_graph(self, component: str, graph_name: str) -> bool:
        """Whether a graph is a separate deployment variant.

        Shared graphs participate in quantization and deployment, so only
        ``_GraphRole.DEPLOY`` variants need observer initialization and encoding
        override from a separate quantization graph.

        Args:
            component: Artifact component key.
            graph_name: Graph key within the component.

        Returns:
            ``True`` when the graph is a separate deployment variant.

        Raises:
            StageError: If the component or graph lacks a role definition.
        """
        if (component_roles := _GRAPH_ROLES.get(component)) is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=f"Component '{component}' has no graph-role definition",
            )
        if (role := component_roles.get(graph_name)) is None:
            raise StageError(
                stage_name=_STAGE_NAME,
                message=f"Component '{component}' has no role for graph '{graph_name}'",
            )
        return role is _GraphRole.DEPLOY

    def _get_quant_io_dtypes(self, quant_recipe: Any) -> Optional[dict]:
        """The ``{"kv_type", "io_type"}`` dtypes a graph's IO is tagged with.

        KV width comes from ``get_kv_io_bit_width`` when available. Output width
        prefers ``get_logits_output_bit_width`` and falls back to
        ``get_act_bit_width`` for recipes without a logits-specific method.
        Unsupported widths are rejected explicitly; 8-bit logits IO is not
        implemented.

        Args:
            quant_recipe: Per-graph recipe instance, if one is configured.

        Returns:
            A map containing available ``kv_type`` and ``io_type`` dtypes, or
            ``None`` when the recipe exposes neither width.

        Raises:
            NotImplementedError: If logits output width is 8 bits.
            RuntimeError: If a reported IO width is unsupported.
        """
        kv_bit_width = (
            quant_recipe.get_kv_io_bit_width()
            if quant_recipe is not None and hasattr(quant_recipe, "get_kv_io_bit_width")
            else None
        )
        io_bit_width = None
        if quant_recipe is not None:
            if hasattr(quant_recipe, "get_logits_output_bit_width"):
                io_bit_width = quant_recipe.get_logits_output_bit_width()
            elif hasattr(quant_recipe, "get_act_bit_width"):
                io_bit_width = quant_recipe.get_act_bit_width()
        width_to_dtype = {8: torch.uint8, 16: torch.uint16}

        if io_bit_width == 8:
            raise NotImplementedError(f"unknown io bit width {io_bit_width}")

        quant_io_dtypes = {}
        for bit_width, dtype_key in (
            (kv_bit_width, "kv_type"),
            (io_bit_width, "io_type"),
        ):
            if bit_width in width_to_dtype:
                quant_io_dtypes[dtype_key] = width_to_dtype[bit_width]
            elif bit_width is not None:
                raise RuntimeError(
                    f"Unsupported quantization IO bit width: {bit_width}"
                )

        return quant_io_dtypes or None
