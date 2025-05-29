from executorch.export.recipe import ExportRecipe, QuantizationRecipe
from executorch.exir import EdgeCompileConfig
from executorch.extension.llm.export.quantizer_lib import get_quantizer_and_quant_params
from executorch.extension.llm.export.export_passes import remove_redundant_transposes
from executorch.extension.llm.export.partitioner_lib import (
    get_coreml_partitioner,
    get_mps_partitioner,
    get_qnn_partitioner,
    get_vulkan_partitioner,
    get_xnnpack_partitioner,
)

def get_llm_recipe(args) -> ExportRecipe:
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        # Force xnnpack to be true if pt2e_quant_params is not None and args.xnnpack is False
        args.xnnpack = True

    quant_recipe = QuantizationRecipe(
        quantizers=quantizers,
    )

    partitioners = []
    if args.xnnpack:
        partitioners.append(get_xnnpack_partitioner(dynamic_quant_only_partitioner=True))
    if args.xnnpack_extended_ops:
        partitioners.append(get_xnnpack_partitioner(dynamic_quant_only_partitioner=False))
    
    return ExportRecipe(
        quantization_recipe=quant_recipe,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
        pre_edge_transform_passes=[remove_redundant_transposes],
        edge_transform_passes=[],
        partitioners=partitioners,
    )
