# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)


def add_input_weight_zp_consts(tosa_graph, node, inputs, output_name):
    """Add input/weight zero-point consts and return their names."""
    input_zp = 0
    if inputs[0].dtype in (ts.DType.INT8, ts.DType.INT16):
        input_qparams = get_input_qparams(node)
        input_zp = input_qparams[0].get_zp_per_tensor()

    weight_zp = 0
    if inputs[1].dtype == ts.DType.INT8:
        input_qparams = get_input_qparams(node)
        zero_points = input_qparams[1].zp
        if isinstance(zero_points, list):
            if not zero_points or any(zp != zero_points[0] for zp in zero_points):
                raise ValueError("TOSA convolution requires one common weight zero point")
            weight_zp = zero_points[0]
        else:
            weight_zp = zero_points

    input_zp_name = f"{output_name}_input_zp"
    weight_zp_name = f"{output_name}_weight_zp"

    input_zp_tensor = tosa_graph.addConst(
        [1], inputs[0].dtype, [input_zp], name=input_zp_name
    )
    weight_zp_tensor = tosa_graph.addConst(
        [1], inputs[1].dtype, [weight_zp], name=weight_zp_name
    )

    return input_zp_tensor.name, weight_zp_tensor.name
