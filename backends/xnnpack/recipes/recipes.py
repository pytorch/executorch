# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from functools import partial
from typing import Any, Callable

from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.export.recipe import ExportRecipe, QuantizationRecipe
from torchao.quantization.quant_api import int8_dynamic_activation_int4_weight


def get_fp32_recipe() -> ExportRecipe:
    return ExportRecipe(
        name="fp32",
        quantization_recipe=None,
        partitioners=[XnnpackPartitioner()],
    )


def get_quant_recipe(quant_recipe_name: str, is_per_channel: bool, is_dynamic: bool, is_qat:bool=False, **_kwargs: Any) -> ExportRecipe:
    # Create quantizer
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(
        is_per_channel=is_per_channel, is_dynamic=is_dynamic, is_qat=is_qat
    )
    quantizer.set_global(operator_config)

    # Create quantization recipe
    quant_recipe = QuantizationRecipe(
        quantizers=[quantizer],
    )

    config_precision = (ConfigPrecisionType.DYNAMIC_QUANT if is_dynamic else ConfigPrecisionType.STATIC_QUANT)

    # Create export recipe
    return ExportRecipe(
        name=quant_recipe_name,
        quantization_recipe=quant_recipe,
        partitioners=[XnnpackPartitioner(config_precision=config_precision)],
        edge_compile_config=get_xnnpack_edge_compile_config(),
    )


def get_8a4w_config(group_size: int = 32) -> ExportRecipe:
    # Create quantization recipe
    quant_recipe = QuantizationRecipe(
        quantizers=None,
        ao_base_config=[
            int8_dynamic_activation_int4_weight(group_size=group_size),
        ],
    )

    # Create export recipe
    return ExportRecipe(
        name="8a4w_quant",
        quantization_recipe=quant_recipe,
        partitioners=[XnnpackPartitioner()],
    )


RECIPE_MAP: dict[str, Callable[..., ExportRecipe]] = {
    "FP32_RECIPE": get_fp32_recipe,
    "QUANT_RECIPE": get_quant_recipe,
    "DYNAMIC_PER_CHANNEL_QUANT_RECIPE": partial(get_quant_recipe, "dynamic_per_channel_quant", is_per_channel=True, is_dynamic=True),
    "STATIC_PER_CHANNEL_QUANT_RECIPE": partial(get_quant_recipe, "static_per_channel_quant", is_per_channel=True, is_dynamic=False),
    "STATIC_PER_TENSOR_QUANT_RECIPE": partial(get_quant_recipe, "static_per_tensor_quant",is_per_channel=False, is_dynamic=False),
    "8A4W_ACCELERATED_RECIPE": get_8a4w_config,
}


def get_xnnpack_recipe(recipe_name: str, **kwargs: Any) -> ExportRecipe:
    assert recipe_name in RECIPE_MAP, f"Recipe {recipe_name} not found."
    return RECIPE_MAP[recipe_name](**kwargs)
