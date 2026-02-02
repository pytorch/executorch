# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops


class QuantConstants:
    # TODO: check keys
    class QUANT_KEY:
        scale = "scales"
        zero_point = "zero_points"
        quant_min = "quant_min"
        quant_max = "quant_max"
        quant_dtype = "quant_dtype"

    PERCHANNEL_KEY_MAP = {
        "scales": QUANT_KEY.scale,
        "zero_points": QUANT_KEY.zero_point,
        "quant_min": QUANT_KEY.quant_min,
        "quant_max": QUANT_KEY.quant_max,
        "dtype": QUANT_KEY.quant_dtype,
    }
    # SNC ir always use key 'scales' and 'zero_points'
    PERTENSOR_KEY_MAP = {
        "scale": QUANT_KEY.scale,
        "zero_point": QUANT_KEY.zero_point,
        "quant_min": QUANT_KEY.quant_min,
        "quant_max": QUANT_KEY.quant_max,
        "dtype": QUANT_KEY.quant_dtype,
    }

    QUANT_OPS_KEY_MAP = {
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default: PERCHANNEL_KEY_MAP,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: PERTENSOR_KEY_MAP,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor: PERTENSOR_KEY_MAP,
    }

    DEQUANT_OPS_KEY_MAP = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: PERTENSOR_KEY_MAP,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor: PERTENSOR_KEY_MAP,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default: PERCHANNEL_KEY_MAP,
    }
