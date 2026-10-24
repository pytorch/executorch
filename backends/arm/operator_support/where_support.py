# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

import torch.fx as fx
from executorch.backends.arm.constants import DQ_OPS
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.specification import keeps_int64
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class WhereSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.aten.where.self]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]

        if len(node.all_input_nodes) != 3:
            self.reporter.report_reject(
                node,
                (
                    "Expected exactly three input nodes, "
                    f"got {len(node.all_input_nodes)} for {node.target}."
                ),
            )
            return False

        condition, x, y = node.all_input_nodes
        if condition.meta["val"].dtype != torch.bool:
            self.reporter.report_reject(
                node,
                f"Type of condition in {node.target} is not torch.bool",
            )
            return False

        x_dtype, y_dtype = x.meta["val"].dtype, y.meta["val"].dtype
        if tosa_spec.support_float():
            supported_float = [
                torch.bool,
                torch.float16,
                torch.float32,
            ]
            if tosa_spec.support_extension("bf16"):
                supported_float.append(torch.bfloat16)

            if x_dtype in supported_float and y_dtype in supported_float:
                return True

        if tosa_spec.support_integer():
            supported_int = [torch.bool, torch.int8, torch.int16, torch.int32]
            # Only a matched int64 pair: MatchArgDtypePass takes the target
            # dtype from the condition and x, so a mixed pair narrows y.
            if keeps_int64(tosa_spec) and x_dtype == y_dtype == torch.int64:
                supported_int.append(torch.int64)

            if (
                x_dtype in supported_int
                or (x_dtype == torch.float32 and x.target in DQ_OPS)
            ) and (
                y_dtype in supported_int
                or (y_dtype == torch.float32 and y.target in DQ_OPS)
            ):
                return True

        self.reporter.report_reject(
            node,
            (
                f"Tensor x dtype {x_dtype} and/or tensor y dtype {y_dtype} is not supported in {node.target} "
                f"for tosa specification {tosa_spec}"
            ),
        )

        return False
