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
    # INT8 Dynamic Quantization
    INT8_DYNAMIC_PER_CHANNEL = "int8_dynamic_per_channel"
    # INT8 Dynamic Activations INT4 Weight Quantization, Axis = 0
    INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_CHANNEL = "int8da_int4w_per_channel"
    # INT8 Dynamic Activations INT4 Weight Quantization, default group_size = 32
    # can be overriden by group_size kwarg
    INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR = "int8da_int4w_per_tensor"
    # INT8 Static Activations INT4 Weight Quantization
    INT8_STATIC_ACT_INT4_WEIGHT_PER_CHANNEL = "int8a_int4w_per_channel"
    INT8_STATIC_ACT_INT4_WEIGHT_PER_TENSOR = "int8a_int44w_per_tensor"
    # INT8 Static Quantization, needs calibration dataset
    INT8_STATIC_PER_CHANNEL = "int8_static_per_channel"
    INT8_STATIC_PER_TENSOR = "int8_static_per_tensor"

    @classmethod
    def get_backend_name(cls) -> str:
        return "xnnpack"
