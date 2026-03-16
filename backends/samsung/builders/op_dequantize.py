# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.samsung.builders.node_visitor import register_node_visitor
from executorch.backends.samsung.builders.op_quantize import _QuantOpVistorBase


# Dequant ops here
@register_node_visitor
class DequantizeVistor(_QuantOpVistorBase):
    target = [
        "quantized_decomposed.dequantize_per_tensor.default",
        "quantized_decomposed.dequantize_per_tensor.tensor",
        "quantized_decomposed.dequantize_per_channel.default",
        "quantized_decomposed.dequantize_per_channel.tensor",
    ]
