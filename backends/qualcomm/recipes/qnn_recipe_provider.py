# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Optional, Sequence

from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.recipes.qnn_recipe_types import (
    QNN_BACKEND,
    QNNRecipeType,
)
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    qnn_edge_config,
)
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
)


class QNNRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return QNN_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(QNNRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create QNN recipe for different precisions and SoC targets"""

        if recipe_type not in self.get_supported_recipes():
            return None

        self._validate_recipe_kwargs(recipe_type, kwargs)

        if recipe_type == QNNRecipeType.FP16:
            return self._build_fp16_recipe(recipe_type, kwargs)

        return None

    def _validate_recipe_kwargs(self, recipe_type: RecipeType, kwargs: Any) -> None:
        """Validate kwargs for each recipe type"""
        expected_keys = self._get_expected_keys(recipe_type)

        unexpected = set(kwargs.keys()) - expected_keys
        if unexpected:
            logging.warning(
                f"QNN Recipe '{recipe_type.value}' received unexpected parameters: {list(unexpected)}, ignoring them"
            )

        self._validate_soc_parameter(kwargs)
        self._validate_partitioner_parameters(kwargs)

    def _get_expected_keys(self, recipe_type: RecipeType) -> set:
        """Get expected parameter keys for a recipe type"""
        _ = recipe_type
        common_keys = {
            "soc_model",
            "skip_node_id_set",
            "skip_node_op_set",
            "skip_mutable_buffer",
        }
        return common_keys

    def _validate_soc_parameter(self, kwargs: Any) -> None:
        """Validate soc_model parameter"""
        if "soc_model" in kwargs:
            soc_model = kwargs["soc_model"]
            if isinstance(soc_model, str):
                try:
                    soc_model = get_soc_to_chipset_map()[soc_model]
                    kwargs["soc_model"] = soc_model
                except KeyError:
                    raise ValueError(
                        f"Invalid SoC model '{soc_model}'. Supported models: {[e.name for e in get_soc_to_chipset_map()]}"
                    )
            elif not isinstance(soc_model, QcomChipset):
                raise ValueError(
                    f"Parameter 'soc_model' must be a QcomChipset enum or string, got {type(soc_model)}"
                )
        else:
            raise ValueError("Parameter 'soc_model' is required")

    def _validate_partitioner_parameters(self, kwargs: Any) -> None:
        """Validate partitioner parameters"""
        if "skip_node_id_set" in kwargs:
            skip_node_id_set = kwargs["skip_node_id_set"]
            if skip_node_id_set is not None and not isinstance(skip_node_id_set, set):
                raise ValueError(
                    f"Parameter 'skip_node_id_set' must be a set or None, got {type(skip_node_id_set)}"
                )

        if "skip_node_op_set" in kwargs:
            skip_node_op_set = kwargs["skip_node_op_set"]
            if skip_node_op_set is not None and not isinstance(skip_node_op_set, set):
                raise ValueError(
                    f"Parameter 'skip_node_op_set' must be a set or None, got {type(skip_node_op_set)}"
                )

        if "skip_mutable_buffer" in kwargs:
            skip_mutable_buffer = kwargs["skip_mutable_buffer"]
            if not isinstance(skip_mutable_buffer, bool):
                raise ValueError(
                    f"Parameter 'skip_mutable_buffer' must be a boolean, got {type(skip_mutable_buffer)}"
                )

    def _build_fp16_recipe(
        self,
        recipe_type: RecipeType,
        kwargs: Any,
    ) -> ExportRecipe:
        soc_model = kwargs["soc_model"]
        skip_node_id_set = kwargs.get("skip_node_id_set", None)
        skip_node_op_set = kwargs.get("skip_node_op_set", None)
        skip_mutable_buffer = kwargs.get("skip_mutable_buffer", False)

        lowering_recipe = self._get_qnn_lowering_recipe(
            use_fp16=True,
            soc_model=soc_model,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            skip_mutable_buffer=skip_mutable_buffer,
        )

        return ExportRecipe(
            name=recipe_type.value,
            aten_transform_passes=[
                lambda method_, ep: QnnPassManager().transform_for_export_pipeline(ep)
            ],
            lowering_recipe=lowering_recipe,
        )

    def _get_qnn_lowering_recipe(
        self,
        use_fp16: bool,
        soc_model: QcomChipset,
        skip_node_id_set: Optional[set] = None,
        skip_node_op_set: Optional[set] = None,
        skip_mutable_buffer: bool = False,
    ) -> LoweringRecipe:
        """Get QNN lowering recipe with optional precision and SoC target"""
        backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)

        compile_specs = generate_qnn_executorch_compiler_spec(
            soc_model=soc_model,
            backend_options=backend_options,
        )

        partitioner = QnnPartitioner(
            compiler_specs=compile_specs,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            skip_mutable_buffer=skip_mutable_buffer,
        )

        edge_compile_config = qnn_edge_config()

        return LoweringRecipe(
            partitioners=[partitioner],
            edge_transform_passes=[
                lambda method_, ep: QnnPassManager().get_to_edge_transform_passes(ep)
            ],
            edge_compile_config=edge_compile_config,
        )
