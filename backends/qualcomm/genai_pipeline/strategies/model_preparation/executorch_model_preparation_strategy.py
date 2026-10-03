# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
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
    """Prepare model artifacts for subsequent ExecuTorch pipeline stages.

    This strategy owns the model-agnostic preparation flow and delegates
    model-family-specific loading to ``ModelLoaderAdapter``. Graph variants share
    one weighted module per component, while their metadata and example inputs stay
    keyed by component and graph because export signatures may differ.

    Example::

        # Text-only
        {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_FORWARD: decoder,
                GRAPH_KV_FORWARD: decoder,
                GRAPH_PREFILL_FORWARD: decoder,
            },
        }

        # Multimodal
        {
            ARTIFACT_TEXT_DECODER: {
                GRAPH_FORWARD: decoder,
                GRAPH_KV_FORWARD: decoder,
                GRAPH_PREFILL_FORWARD: decoder,
            },
            ARTIFACT_TOK_EMBEDDING: {
                GRAPH_FORWARD: tok_embedding,
                ...,
            },
            ARTIFACT_VISION_ENCODER: {
                GRAPH_FORWARD: vision_encoder,
            },
            ARTIFACT_AUDIO_ENCODER: {
                GRAPH_FORWARD: audio_encoder,
            },
        }

    The preparation flow mirrors :meth:`invoke`:

    1. Load one weighted module per component and graph variant.
    2. Read per-graph metadata.
    3. Build model-native example inputs for every graph variant.
    4. Select one weight-sharing module per component.
    5. Apply component-level module transforms.
    6. Load the tokenizer.
    7. Build the model-specific inference helper used by quantization, if any.
    8. Optionally export the tokenizer for runtime.
    9. Extract the tokenizer chat template, falling back to ``extra_options``
       when the tokenizer has none.

    Args:
        model_loader_adapter: Injectable adapter for model and tokenizer
            loading. Defaults to ``DefaultModelLoaderAdapter`` if not provided.
    """

    def __init__(
        self,
        model_loader_adapter: Optional[ModelLoaderAdapter] = None,
    ) -> None:
        if model_loader_adapter is None:
            from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.default_model_loader_adapter import (
                DefaultModelLoaderAdapter,
            )

            model_loader_adapter = DefaultModelLoaderAdapter()
        self._adapter = model_loader_adapter

    @property
    def adapter(self) -> ModelLoaderAdapter:
        """The model loader adapter used by this strategy."""
        return self._adapter

    def invoke(
        self,
        context: PipelineContext,
        input_config: ModelPreparationInputConfig,
    ) -> ModelPreparationOutputConfig:
        """Prepare component modules and runtime metadata for later pipeline stages.

        Args:
            context: Pipeline context. Its artifact directory is used when
                exporting the runtime tokenizer.
            input_config: Model identity, target SoC, and preparation inputs.

        Configuration:
            ``input_config.extra_options["model_options"]`` configures model
            loading:

            - ``model_arch``: Component- and graph-keyed module constructors.
            - ``state_dict_loader``: Component-keyed loaders. Each loader
              accepts a Hugging Face ``repo_id`` and returns a state dict.
            - ``weight_transforms``: Component-keyed transforms applied while
              loading weights.
            - ``module_transforms``: Component-keyed transforms applied after
              one shared module is selected for each component.
            - ``num_shardings``: Optional component shard counts.

            Root ``input_config.extra_options`` configures tokenizer and runtime
            output:

            - ``tokenizer_options``: Options passed to ``load_tokenizer``.
            - ``export_tokenizer``: Enables runtime tokenizer export.
            - ``tokenizer_export_options``: Options passed when tokenizer
              export is enabled.
            - ``chat_template``: Fallback used only when the loaded tokenizer
              has no chat template.

        Returns:
            A ``ModelPreparationOutputConfig`` with component-level modules,
            component- and graph-keyed inputs and metadata, tokenizer state,
            optional quantization inference support, and shard counts.

        Raises:
            StageError: If required fields are missing or any preparation step
                fails.
        """
        logger.info(
            "Starting model preparation for '%s' on SoC=%s",
            input_config.model_name,
            input_config.soc_model,
        )

        self._validate_input(input_config)

        try:
            extra = dict(input_config.extra_options)
            model_options = extra.get("model_options", {})

            # Step 1: Load model
            logger.debug("Loading model")
            modules = self._adapter.load_model(
                model_name=input_config.model_name,
                extra_options=model_options,
            )

            # Step 2: Get per-graph constant metadata.
            # Text decoder metadata determines whether its nested example inputs
            # include KV-cache arguments.
            meta = self._get_metadata(modules)

            # Step 3: Build per-graph export example inputs from the model.
            # These are deliberately *not* taken from the calibration dataset:
            # they define the exported graph's positional signature (including
            # zero-initialized KV caches, which no dataset sample carries) and
            # the dataset's own attention-mask schema is derived from them.
            logger.debug("Building example inputs for export")
            example_inputs = self._get_example_inputs(modules)

            # Select one module per component.
            # TODO: Make module-wrapper ``get_example_inputs`` accept shape-related
            # parameters so different-shaped inputs are derived from the shape request
            # rather than from separate graph-wrapper modules.
            model_module = self._get_component_module(modules)
            del modules
            gc.collect()

            # Step 4: Apply component-level transforms.
            logger.debug("Applying module transforms")
            model_module = self._apply_module_transforms(
                model_module,
                extra_options=model_options,
            )

            # Step 5: Load tokenizer.
            logger.debug("Loading tokenizer")
            tokenizer = self._adapter.load_tokenizer(
                model_name=input_config.model_name,
                extra_options=extra.get("tokenizer_options"),
            )

            # Step 6: Create the model-specific inference instance or callable used
            # by quantization. Returns None when not applicable.
            inference = self._adapter.get_inference(
                meta,
                example_inputs,
                extra_options=extra,
            )

            # Step 7: Optionally export tokenizer for runtime
            runtime_tokenizer_path = None
            if extra.get("export_tokenizer", False):
                logger.debug("Exporting tokenizer for runtime use")
                output_dir = Path(context.artifact_dir) / "tokenizer"
                runtime_tokenizer_path = self._adapter.export_tokenizer(
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    extra_options=extra.get("tokenizer_export_options"),
                )

            # Step 8: Extract chat_template from tokenizer (for instruct models).
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
                runtime_tokenizer_path=runtime_tokenizer_path,
                chat_template=chat_template,
                meta=meta,
                inference=inference,
                num_shardings=model_options.get(
                    "num_shardings",
                    extra.get("num_shardings"),
                ),
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

    def _get_metadata(self, modules: Dict[str, Dict[str, Any]]) -> Dict:
        """Collect non-empty constant metadata for every component graph.

        Args:
            modules: Component- and graph-keyed loaded modules.

        Returns:
            Component- and graph-keyed metadata, omitting graphs and components
            whose adapter metadata is empty.
        """
        meta = {}
        for component, graph_modules in modules.items():
            component_meta = {
                graph_name: graph_meta
                for graph_name, graph_module in graph_modules.items()
                if (graph_meta := self._adapter.get_metadata(graph_module))
            }
            if component_meta:
                meta[component] = component_meta
        return meta

    def _get_example_inputs(
        self,
        modules: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Collect model-native example inputs for every component graph.

        Args:
            modules: Component- and graph-keyed loaded modules.

        Returns:
            Component- and graph-keyed example inputs that define each graph's
            export signature.
        """
        example_inputs = {}
        for component, graph_modules in modules.items():
            example_inputs[component] = {}
            for graph_name, graph_module in graph_modules.items():
                logger.debug(
                    "Building example inputs for component '%s' graph '%s'",
                    component,
                    graph_name,
                )
                example_inputs[component][graph_name] = (
                    self._adapter.get_example_inputs(graph_module)
                )
        return example_inputs

    def _apply_module_transforms(
        self,
        modules: Dict[str, Any],
        extra_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply configured transforms to each selected component module.

        Args:
            modules: Component-keyed modules selected from graph variants.
            extra_options: The ``model_options`` map containing component-keyed
                ``module_transforms``.

        Returns:
            Component-keyed modules after their configured transforms run.

        Raises:
            ValueError: If ``module_transforms`` is not component-keyed.
        """
        module_transforms = extra_options.get("module_transforms", {})
        if not isinstance(module_transforms, dict):
            raise ValueError("module_transforms must be component-keyed")
        return {
            component: self._adapter.apply_module_transforms(
                module,
                module_transforms=module_transforms.get(component, []),
            )
            for component, module in modules.items()
        }

    def _get_component_module(
        self, modules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select one shared module per component.

        Graph variants share a component's module and weights, while their
        graph-specific example inputs and metadata remain separate.

        Args:
            modules: Component- and graph-keyed loaded modules.

        Returns:
            Component-keyed representative modules, using the first graph
            variant for each non-empty component.
        """
        model_module = {}
        for component, graph_modules in modules.items():
            if not graph_modules:
                continue

            selected_graph_name = next(iter(graph_modules))
            model_module[component] = graph_modules[selected_graph_name]

        return model_module
