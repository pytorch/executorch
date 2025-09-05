# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class IndexSelectSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.index_select.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]

        weights_shape = node.all_input_nodes[0].meta["val"].shape
        indices_val = node.all_input_nodes[1].meta["val"]
        indices_dtype = indices_val.dtype

        if indices_dtype != torch.int32:
            self.reporter.report_reject(
                node,
                f"Indices dtype {indices_val.dtype} is not supported in {node.target}.",
            )
            return False

        if not (
            len(weights_shape) == 2
            or (len(weights_shape) == 3 and weights_shape[0] == 1)
        ):
            self.reporter.report_reject(
                node, f"{node.target} with weights shape {weights_shape} not supported."
            )
            return False
        return True
