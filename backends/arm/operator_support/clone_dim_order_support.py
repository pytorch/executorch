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

        supported_dtypes = {torch.bool, torch.int8, torch.int16, torch.int32}
        if tosa_spec.support_float():
            supported_dtypes |= {torch.bfloat16, torch.float16, torch.float32}

        # Check input type
        assert len(node.all_input_nodes) == 1
        input_val = node.all_input_nodes[0].meta["val"]
        assert isinstance(input_val, torch._subclasses.FakeTensor)
        input_dtype = input_val.dtype
        if input_dtype not in supported_dtypes:
            self.reporter.report_reject(
                node,
                f"Input dtype {input_val.dtype} is not supported in {node.target}.",
            )
            return False

        # Check output type
        output_val = node.meta["val"]
        assert isinstance(output_val, torch._subclasses.FakeTensor)
        if output_val.dtype != input_dtype:
            self.reporter.report_reject(
                node,
                f"Input dtype {input_val.dtype} does not match {output_val.dtype}.",
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
