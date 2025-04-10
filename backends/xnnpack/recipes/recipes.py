# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Any, Callable

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    duplicate_dynamic_quant_chain_pass,
    DuplicateDynamicQuantChainPass,
)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import ExportRecipe

def get_generic_fp32_cpu_recipe() -> ExportRecipe:
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    return ExportRecipe(
       name = "fp32_recipe",
       quantizer = None,
       partitioners=[XnnpackPartitioner()],

    )

def get_dynamic_quant_recipe() -> ExportRecipe:
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(
        is_per_channel=True, is_dynamic=True
    )
    quantizer.set_global(operator_config)
    DuplicateDynamicQuantChainPass
    return ExportRecipe(
       name = "dynamic_quant_recipe",
       quantizer = quantizer,
       partitioners=[XnnpackPartitioner()],
       pre_edge_transform_passes=duplicate_dynamic_quant_chain_pass,
    )

RECIPE_MAP: dict[str, Callable[[], ExportRecipe]] = {
    "FP32_CPU_ACCELERATED_RECIPE": get_generic_fp32_cpu_recipe,
    "DYNAMIC_QUANT_CPU_ACCELERATED_RECIPE": get_dynamic_quant_recipe,
}

def get_xnnpack_recipe(recipe_name:str, **kwargs: Any) -> ExportRecipe:
    assert recipe_name in RECIPE_MAP, f"Recipe {recipe_name} not found."
    return RECIPE_MAP[recipe_name](**kwargs)
