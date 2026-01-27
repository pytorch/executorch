# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``aten.slice_copy`` in TOSA.

Rely on preprocessing (e.g. DecomposeStridedSliceCopyPass) to rewrite any
non-unit-step slicing into supported ops. Assume static shapes and constant
slicing parameters.

Check:
- args length is 4 or 5
- If present, require step > 0.
- Require dtype compatible with the selected TOSA profile (allow bool in both).

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
class SliceCopySupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``aten.slice_copy``."""

    targets = [exir_ops.edge.aten.slice_copy.Tensor]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        if len(node.args) not in (4, 5):
            self.reporter.report_reject(
                node,
                f"{node.target}: expected 4 or 5 args, got {len(node.args)}.",
            )
            return False

        if len(node.args) == 5:
            step = node.args[4]
            if step <= 0:  # type: ignore[operator]
                self.reporter.report_reject(
                    node,
                    f"{node.target}: step must be > 0, got {step}.",
                )
                return False

        values_dtype = node.args[0].meta["val"].dtype  # type: ignore[union-attr]

        SUPPORTED_INT_DTYPES = (torch.int8, torch.int16, torch.int32)
        SUPPORTED_FLOAT_DTYPES = (torch.float16, torch.float32)
        SUPPORTED_DTYPES = (torch.bool,) + SUPPORTED_INT_DTYPES + SUPPORTED_FLOAT_DTYPES

        # bool is supported in both INT and FP profiles
        if values_dtype == torch.bool:
            return True
        # ints require INT profile
        elif values_dtype in SUPPORTED_INT_DTYPES:
            if not tosa_spec.support_integer():
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires INT profile.",
                )
                return False

        # fp16/fp32: either FP profile, or INT profile (via quantization)
        elif values_dtype in SUPPORTED_FLOAT_DTYPES:
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
                f"expected one of {SUPPORTED_DTYPES}.",
            )
            return False

        return True
