# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for dim-order clone in TOSA.

This module registers a support check for ``dim_order_ops._clone_dim_order``
ensuring input/output dtypes match and the value types are FakeTensors.

"""

import logging

import torch
import torch.fx as fx
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


@register_tosa_support_check
class CloneSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``_clone_dim_order``."""

    targets = [exir_ops.edge.dim_order_ops._clone_dim_order.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if the node is supported by TOSA.

        Verify the operator target, the number and types of inputs/outputs, and
        check that input and output dtypes match.

        """
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

        return True
