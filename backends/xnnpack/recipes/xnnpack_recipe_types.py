# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


class XNNPackRecipeType(RecipeType):
    """XNNPACK-specific recipe types"""

    FP32 = "fp32"

    ## PT2E-based quantization recipes
    # INT8 Dynamic Quantization
    PT2E_INT8_DYNAMIC_PER_CHANNEL = "pt2e_int8_dynamic_per_channel"
    # INT8 Static Quantization, needs calibration dataset
    PT2E_INT8_STATIC_PER_CHANNEL = "pt2e_int8_static_per_channel"
    PT2E_INT8_STATIC_PER_TENSOR = "pt2e_int8_static_per_tensor"

    ## TorchAO-based quantization recipes
    # INT8 Dynamic Activations INT4 Weight Quantization, Axis = 0
    TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_CHANNEL = (
        "torchao_int8da_int4w_per_channel"
    )
    # INT8 Dynamic Activations INT4 Weight Quantization, default group_size = 32
    # can be overriden by group_size kwarg
    TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR = "torchao_int8da_int4w_per_tensor"

    @classmethod
    def get_backend_name(cls) -> str:
        return "xnnpack"
