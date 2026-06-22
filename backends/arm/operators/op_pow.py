# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import register_node_visitor
from executorch.backends.arm.operators.simple_node_visitor import (
    SimpleNodeVisitor,
    SimpleNodeVisitorConfig,
)
from executorch.backends.arm.tosa import TosaSpecification

FP_SPECS = TosaSpecification.all_versions_for_profile("FP")


@register_node_visitor
class PowVisitor(SimpleNodeVisitor):
    target = "aten.pow.Tensor_Tensor"
    tosa_specs = FP_SPECS

    @classmethod
    def get_config(cls) -> SimpleNodeVisitorConfig:
        return SimpleNodeVisitorConfig(
            tosa_op=ts.Op.POW,
            attr_method="PowAttribute",
            num_inputs=2,
            input_dtypes=[ts.DType.FP16, ts.DType.FP32, ts.DType.BF16],
        )
