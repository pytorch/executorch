# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


from executorch.export import RecipeType


COREML_BACKEND: str = "coreml"


class CoreMLRecipeType(RecipeType):
    """CoreML-specific generic recipe types"""

    ## All the recipes accept common kwargs
    # 1. minimum_deployment_unit (default: None)
    # 2. compute_unit (default: ct.ComputeUnit.ALL)

    # FP32 precision recipe, defaults to values published by the CoreML backend and partitioner
    FP32 = "coreml_fp32"

    # FP16 precision recipe, defaults to values published by the CoreML backend and partitioner
    FP16 = "coreml_fp16"

    ## PT2E-based quantization recipes
    # INT8 Static Quantization (weights + activations), requires calibration dataset
    PT2E_INT8_STATIC = "coreml_pt2e_int8_static"
    # INT8 Weight-only Quantization (activations remain FP32)
    PT2E_INT8_WEIGHT_ONLY = "coreml_pt2e_int8_weight_only"

    ## TorchAO-based quantization recipes
    # All TorchAO recipes accept filter_fn kwarg to control which layers are quantized
    # INT4 Weight-only Quantization, per-channel (axis=0)
    # Additional kwargs: filter_fn (default: Embedding and linear layers)
    TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL = "coreml_torchao_int4_weight_only_per_channel"
    # INT4 Weight-only Quantization, per-group
    # Additional kwargs: group_size (default: 32), filter_fn (default: Embedding and linear layers)
    TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP = "coreml_torchao_int4_weight_only_per_group"
    # INT8 Weight-only Quantization, per-channel (axis=0)
    # Additional kwargs: filter_fn (default: Embedding and linear layers)
    TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL = "coreml_torchao_int8_weight_only_per_channel"
    # INT8 Weight-only Quantization, per-group
    # Additional kwargs: group_size (default: 32), filter_fn (default: Embedding and linear layers)
    TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP = "coreml_torchao_int8_weight_only_per_group"

    ## Codebook/Palettization Quantization
    # Additional mandatory kwargs: bits (range: 1-8), block_size (list of ints),
    # filter_fn (default: targets Linear and Embedding layers)
    CODEBOOK_WEIGHT_ONLY = "coreml_codebook_weight_only"

    @classmethod
    def get_backend_name(cls) -> str:
        return COREML_BACKEND
