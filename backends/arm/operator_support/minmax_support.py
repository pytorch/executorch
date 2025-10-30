# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class MinMaxSupported(SupportedTOSAOperatorCheck):
    targets = [
        exir_ops.edge.aten.max.dim,
        exir_ops.edge.aten.min.dim,
    ]

    # TODO : "MLETORCH-718 : Quantization of indices in arm_quantizer"
    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(self, node: fx.Node, tosa_spec: TosaSpecification):
        if node.target in [exir_ops.edge.aten.max.dim, exir_ops.edge.aten.min.dim]:
            no_argmax = len(node.users) == 1
            no_argmax_users = (len(node.users) == 2) and (
                len(list(node.users)[1].users) == 0
            )

            if not (no_argmax or no_argmax_users):
                self.reporter.report_reject(
                    node,
                    (
                        "Using the indices output is not supported; only usage of the "
                        "values output is supported."
                    ),
                )
                return False

        return True
