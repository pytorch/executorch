# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


@register_tosa_support_check
class CloneSupported(SupportedTOSAOperatorCheck):
    targets = [exir_ops.edge.dim_order_ops._clone_dim_order.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        if node.target not in self.targets:
            self.reporter.report_reject(node, f"Target {node.target} is not supported.")
            return False

        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            self.reporter.report_reject(node, "Non tensor clones are not supported")
            return False

        # Check input node
        if len(node.all_input_nodes) != 1:
            self.reporter.report_reject(
                node, f"Expected 1 input node, got {len(node.all_input_nodes)}"
            )
            return False

        input_val = node.all_input_nodes[0].meta["val"]
        if not isinstance(input_val, torch._subclasses.FakeTensor):
            self.reporter.report_reject(node, "Expected input to be a FakeTensor.")
            return False

        input_dtype = input_val.dtype

        # Check output node
        output_val = node.meta["val"]
        if not isinstance(output_val, torch._subclasses.FakeTensor):
            self.reporter.report_reject(node, "Expected output to be a FakeTensor.")
            return False

        if output_val.dtype != input_dtype:
            self.reporter.report_reject(
                node,
                f"Input dtype {input_val.dtype} does not match {output_val.dtype}.",
            )
            return False

        # Check memory format
        if "memory_format" in node.kwargs:
            if node.kwargs["memory_format"] in (torch.preserve_format,):
                self.reporter.report_reject(
                    node,
                    f"Argument 'memory_format' is not supported for "
                    f"{node.target} right now.",
                )
                return False

        # Check dim_order
        if "dim_order" in node.kwargs:
            dim_order = node.kwargs["dim_order"]
            # pyre-ignore[6]
            if dim_order != list(range(len(dim_order))):  # type: ignore[arg-type]
                self.reporter.report_reject(
                    node,
                    f"Argument {dim_order=} is not supported for "
                    f"{node.target} right now.",
                )
                return False

        return True
