# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Declare operator support for ``edge.aten.unfold_copy`` in TOSA.

This support check matches the subset intended to be handled by the Arm
unfold lowering (e.g. DecomposeUnfoldToGatherPass), similar to how
GatherSupported gates CanonicalizeGatherPass.

Supported pattern:

- target: exir_ops.edge.aten.unfold_copy.default
- args: exactly (x, dimension, size, step)  (i.e. len(node.args) == 4)
- x must be rank >= 1 (e.g. [T], [B,T], [B,T,F], ...)
- dimension may be negative (normalized against rank)
- size and step must be > 0
- the selected dimension length must satisfy D >= size

Notes:
- Dtype/profile constraints apply (see dtype checks below).
- This check assumes static shapes and constant dim/size/step.
"""

import torch
import torch.fx as fx

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class UnfoldCopySupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``edge.aten.unfold_copy``."""

    targets = [exir_ops.edge.aten.unfold_copy.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        """Return True if the node is supported by TOSA."""

        if len(node.args) != 4:
            self.reporter.report_reject(
                node,
                f"{node.target}: expected 4 args (x, dimension, size, step), "
                f"got {len(node.args)}.",
            )
            return False

        x_arg, dim, size, step = node.args

        if size <= 0 or step <= 0:  # type: ignore[operator]
            self.reporter.report_reject(
                node,
                f"{node.target}: size and step must be > 0, got size={size} "
                f"step={step}.",
            )
            return False

        x_val = x_arg.meta["val"]  # type: ignore[union-attr]
        x_shape = tuple(x_val.shape)
        x_rank = len(x_shape)

        if x_rank < 1:
            self.reporter.report_reject(
                node,
                f"{node.target}: input must be rank>=1, got shape={x_shape}.",
            )
            return False

        # Values dtype validation
        values_dtype = x_val.dtype
        # ints (and bool via casts) require INT profile
        if values_dtype in (torch.bool, torch.int8, torch.int16, torch.int32):
            if not tosa_spec.support_integer():
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires INT profile.",
                )
                return False
        # fp16/fp32: either FP profile, or INT profile (via quantization)
        elif values_dtype in (torch.float16, torch.float32):
            if not (tosa_spec.support_float() or tosa_spec.support_integer()):
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires FP profile or "
                    "INT profile (with quantization).",
                )
                return False
        else:
            self.reporter.report_reject(
                node,
                f"{node.target}: unsupported values dtype {values_dtype}; "
                "expected bool/int8/int16/int32/float16/float32.",
            )
            return False

        # Basic validity on unfolded dimension
        dim_norm = dim % x_rank  # type: ignore[operator]
        D = x_shape[dim_norm]  # type: ignore[index]
        if D < size:
            self.reporter.report_reject(
                node,
                f"{node.target}: invalid unfold (x.shape[{dim_norm}]={D} < "
                f"size={size}) for shape={x_shape}.",
            )
            return False

        return True
