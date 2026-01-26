# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Declare operator support for ``edge.aten.gather`` in TOSA.

This support check matches the subset accepted by CanonicalizeGatherPass:

- target: exir_ops.edge.aten.gather.default
- args: exactly (x, dim, index)  (i.e. len(node.args) == 3)
- dim must map to 1
- x must be rank-2 or rank-3
- index must be rank-2 or rank-3.
- for rank-3 x.shape[-1] must match index.shape[-1]
- index dtype must be int32
- batch dim must match: x.shape[0] == index.shape[0]

Dtype gating is capability-based:

- int8/int16/int32 values require INT profile.
- bool values require INT profile (handled via casts: bool -> int8 -> bool).
- fp16/fp32 values are supported via FP profile directly, or via quantization
  when running under an INT profile.

Note:
- For 2D inputs CanonicalizeGatherPass reshapes values to [N, K, 1] and keeps
  indices as [N, W],then lowers via the TOSA gather dialect.
- For 3D inputs CanonicalizeGatherPass permutes and reshapes values and indices
  to [N*C, K, 1] and [N*C, W] respectively, then lowers via the TOSA gather dialect.
"""

from typing import cast

import torch
import torch.fx as fx

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class GatherSupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``edge.aten.gather``."""

    targets = [exir_ops.edge.aten.gather.default]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        if len(node.args) != 3:
            self.reporter.report_reject(
                node,
                f"{node.target}: expected 3 args (x, dim, index), got "
                f"{len(node.args)}.",
            )
            return False

        x_arg, dim, index_arg = node.args[0], node.args[1], node.args[2]
        x_val = x_arg.meta["val"]  # type: ignore[union-attr]
        index_val = index_arg.meta["val"]  # type: ignore[union-attr]

        x_shape = tuple(x_val.shape)
        index_shape = tuple(index_val.shape)
        dim = cast(int, dim)
        dim = dim % len(x_shape)

        # ---- index dtype ----
        if index_val.dtype != torch.int32:
            self.reporter.report_reject(
                node,
                f"{node.target}: index dtype {index_val.dtype} not supported; "
                "expected int32.",
            )
            return False

        # ---- dim + rank ----
        if not ((dim == 1) and len(x_shape) in (2, 3) and len(index_shape) in (2, 3)):
            self.reporter.report_reject(
                node,
                f"{node.target}: unsupported dim/rank; got {dim=}, "
                f"x_rank={len(x_shape)}, index_rank={len(index_shape)}; "
                "supported: dim in {1,} with rank-2/3 x and rank-2/3 index.",
            )
            return False

        if len(index_shape) == 3:
            if x_shape[-1] != index_shape[-1]:
                self.reporter.report_reject(
                    node,
                    f"{node.target}: trailing dimension size mismatch "
                    f"{x_shape[-1]=} vs {index_shape[-1]=}.",
                )
                return False

        # ---- batch dim compatibility ----
        if x_shape[0] != index_shape[0]:
            self.reporter.report_reject(
                node,
                f"{node.target}: batch mismatch {x_shape[0]=} vs {index_shape[0]=}.",
            )
            return False

        # ---- values dtype ----
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
