# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
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
class CloneDimOrderSupport(SupportedTOSAOperatorCheck):
    targets = [
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        assert node.target in self.targets

        # Check input type
        assert len(node.all_input_nodes) == 1
        input_val = node.all_input_nodes[0].meta["val"]
        assert isinstance(input_val, torch._subclasses.FakeTensor)
        input_dtype = input_val.dtype

        # Check output type
        output_val = node.meta["val"]
        assert isinstance(output_val, torch._subclasses.FakeTensor)
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
