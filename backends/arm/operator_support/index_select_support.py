# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``aten.index_select`` in TOSA.

This support check is intentionally broad to enable "full" index_select support
via decomposition to backend TOSA gather.

Constraints:
- target: exir_ops.edge.aten.index_select.default
- args: exactly (input, dim, index)
- input rank must be >= 1 and dtype compatible with the active TOSA spec
- index must be rank-1 and dtype int32
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
class IndexSelectSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``aten.index_select``."""

    targets = [exir_ops.edge.aten.index_select.default]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        if len(node.args) != 3:
            self.reporter.report_reject(
                node,
                f"{node.target}: expected 3 args (input, dim, index), "
                f"got {len(node.args)}.",
            )
            return False

        x_arg, _, index_arg = node.args

        x_val = x_arg.meta["val"]  # type: ignore[union-attr]
        index_val = index_arg.meta["val"]  # type: ignore[union-attr]

        x_shape = tuple(x_val.shape)
        x_rank = len(x_shape)

        if x_rank < 1:
            self.reporter.report_reject(
                node,
                f"{node.target}: input must be rank>=1, got shape={x_shape}.",
            )
            return False

        # Indices must be int32 and rank-1.
        if not (index_val.dtype == torch.int32 and index_val.dim() == 1):
            self.reporter.report_reject(
                node,
                f"{node.target}: index must be rank-1 int32, got dtype={index_val.dtype} "
                f"shape={tuple(index_val.shape)}.",
            )
            return False

        # Dtype constraints
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

        return True
