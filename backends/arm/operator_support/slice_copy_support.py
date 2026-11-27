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
from executorch.backends.arm.operator_support.dtype_support import (
    is_indexing_dtype_supported,
)
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

    @staticmethod
    def _has_symbolic_bound(node: fx.Node) -> bool:
        return any(
            isinstance(bound, torch.SymInt)
            or isinstance(bound, fx.Node)
            and isinstance(bound.meta.get("val"), torch.SymInt)
            for bound in node.args[2:4]
        )

    @staticmethod
    def _has_no_non_positive_static_output_size(node: fx.Node) -> bool:
        return all(
            not isinstance(output_size, int) or output_size > 0
            for output_size in node.meta["val"].shape
        )

    def _has_valid_slice_arguments(self, node: fx.Node) -> bool:
        if len(node.args) not in (4, 5):
            self.reporter.report_reject(
                node,
                f"{node.target}: expected 4 or 5 args, got {len(node.args)}.",
            )
            return False
        if len(node.args) == 5 and node.args[4] <= 0:  # type: ignore[operator]
            self.reporter.report_reject(
                node, f"{node.target}: step must be > 0, got {node.args[4]}."
            )
            return False
        return True

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        if not self._has_valid_slice_arguments(node):
            return False

        if self._has_symbolic_bound(node):
            self.reporter.report_reject(
                node, "Symbolic slice bounds cannot be lowered to TOSA SLICE."
            )
            return False

        if not self._has_no_non_positive_static_output_size(node):
            self.reporter.report_reject(
                node, "TOSA SLICE requires a guaranteed positive output size."
            )
            return False

        values_dtype = node.args[0].meta["val"].dtype  # type: ignore[union-attr]

        # bool is supported in both INT and FP profiles
        if values_dtype == torch.bool:
            return True
        if (
            values_dtype == torch.int64
            and tosa_spec.support_integer()
            and tosa_spec.support_extension("int64")
        ):
            return True

        return is_indexing_dtype_supported(node, values_dtype, tosa_spec, self.reporter)
