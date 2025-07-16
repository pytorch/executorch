# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional, Sequence

import torch

from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from executorch.backends.xnnpack.recipes.xnnpack_recipe_types import XNNPackRecipeType
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_edge_compile_config,
    get_xnnpack_executorch_backend_config,
)
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    QuantizationRecipe,
    RecipeType,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig


class XNNPACKRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return "xnnpack"

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(XNNPackRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create XNNPACK recipe"""

        if recipe_type not in self.get_supported_recipes():
            return None

        # Validate kwargs
        self._validate_recipe_kwargs(recipe_type, **kwargs)

        if recipe_type == XNNPackRecipeType.FP32:
            return self._build_fp32_recipe(recipe_type)

        elif recipe_type == XNNPackRecipeType.INT8_DYNAMIC_PER_CHANNEL:
            return self._build_quantized_recipe(
                recipe_type, is_per_channel=True, is_dynamic=True
            )

        elif recipe_type == XNNPackRecipeType.INT8_DYNAMIC_PER_TENSOR:
            return self._build_quantized_recipe(
                recipe_type, is_per_channel=False, is_dynamic=True
            )

        elif recipe_type == XNNPackRecipeType.INT8_STATIC_PER_CHANNEL:
            return self._build_quantized_recipe(
                recipe_type, is_per_channel=True, is_dynamic=False
            )

        elif recipe_type == XNNPackRecipeType.INT8_STATIC_PER_TENSOR:
            return self._build_quantized_recipe(
                recipe_type, is_per_channel=False, is_dynamic=False
            )

        elif recipe_type == XNNPackRecipeType.INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_CHANNEL:
            return self._build_int8da_intx_weight_recipe(
                recipe_type=recipe_type,
                is_per_channel=True,
                weight_dtype=torch.int4,
            )

        elif recipe_type == XNNPackRecipeType.INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR:
            group_size = kwargs.get("group_size", 32)
            return self._build_int8da_intx_weight_recipe(
                recipe_type=recipe_type,
                is_per_channel=False,
                weight_dtype=torch.int4,
                group_size=group_size,
            )
        return None

    def _build_fp32_recipe(self, recipe_type: RecipeType) -> ExportRecipe:
        return ExportRecipe(
            name=recipe_type.value,
            edge_compile_config=get_xnnpack_edge_compile_config(),
            executorch_backend_config=get_xnnpack_executorch_backend_config(),
            partitioners=[XnnpackPartitioner()],
        )

    def _build_quantized_recipe(
        self,
        recipe_type: RecipeType,
        is_per_channel: bool = True,
        is_dynamic: bool = True,
        is_qat: bool = False,
    ) -> ExportRecipe:
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=is_per_channel, is_dynamic=is_dynamic, is_qat=is_qat
        )
        quantizer.set_global(operator_config)

        quant_recipe = QuantizationRecipe(quantizers=[quantizer])

        precision_type = (
            ConfigPrecisionType.DYNAMIC_QUANT
            if is_dynamic
            else ConfigPrecisionType.STATIC_QUANT
        )

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quant_recipe,
            edge_compile_config=get_xnnpack_edge_compile_config(),
            executorch_backend_config=get_xnnpack_executorch_backend_config(),
            partitioners=[XnnpackPartitioner(config_precision=precision_type)],
        )

    def _build_int8da_intx_weight_recipe(
        self,
        recipe_type: RecipeType,
        is_per_channel: bool = True,
        weight_dtype: torch.dtype = torch.int4,
        group_size: int = 32,
    ) -> ExportRecipe:
        if is_per_channel:
            weight_granularity = PerAxis(axis=0)
        else:
            weight_granularity = PerGroup(group_size=group_size)

        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            weight_granularity=weight_granularity,
        )

        quant_recipe = QuantizationRecipe(
            quantizers=None,
            ao_base_config=[config],
        )

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=quant_recipe,
            edge_compile_config=get_xnnpack_edge_compile_config(),
            executorch_backend_config=get_xnnpack_executorch_backend_config(),
            partitioners=[XnnpackPartitioner()],
        )

    def _validate_recipe_kwargs(self, recipe_type: RecipeType, **kwargs: Any) -> None:
        if recipe_type == XNNPackRecipeType.INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR:
            expected_keys = {"group_size"}
            unexpected = set(kwargs.keys()) - expected_keys
            if unexpected:
                raise ValueError(
                    f"Recipe '{recipe_type.value}' only accepts 'group_size' parameter. "
                    f"Unexpected parameters: {list(unexpected)}"
                )
            if "group_size" in kwargs:
                group_size = kwargs["group_size"]
                if not isinstance(group_size, int):
                    raise ValueError(
                        f"Parameter 'group_size' must be an integer, got {type(group_size).__name__}: {group_size}"
                    )
        elif kwargs:
            # All other recipes don't expect any kwargs
            unexpected = list(kwargs.keys())
            raise ValueError(
                f"Recipe '{recipe_type.value}' does not accept any parameters. "
                f"Unexpected parameters: {unexpected}"
            )
