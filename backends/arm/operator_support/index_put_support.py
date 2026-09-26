# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``aten.index_put``."""

from typing import cast

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)

from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class IndexPutSupported(SupportedTOSAOperatorCheck):
    """Reject unsupported ``index_put`` cases.

    Explicit integer indices are fully supported.

    For boolean mask, there are limitations:
    - boolean index cases only supports one bool index
    - boolean index cases must use a scalar ``values`` tensor
    - boolean index cases don't support accumulate = True.

    """

    targets = [exir_ops.edge.aten.index_put.default]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        indices_tensors = cast(list[fx.Node], node.args[1])

        # None indexes mean "select whole dim", we can handle that.
        explicit_indices = [index for index in indices_tensors if index is not None]
        has_bool_index = any(
            get_first_fake_tensor(index).dtype == torch.bool
            for index in explicit_indices
        )
        has_non_bool_index = any(
            get_first_fake_tensor(index).dtype != torch.bool
            for index in explicit_indices
        )

        if has_bool_index and has_non_bool_index:
            self.reporter.report_reject(
                node,
                (
                    "Mixed boolean mask and integer indices in "
                    "index_put are not supported."
                ),
            )
            return False

        if has_bool_index and len(explicit_indices) != 1:
            self.reporter.report_reject(
                node,
                "Boolean mask index_put only supports a single explicit bool index.",
            )
            return False

        if has_bool_index:
            values = cast(fx.Node, node.args[2])
            values_tensor = get_first_fake_tensor(values)
            if values_tensor.numel() != 1:
                self.reporter.report_reject(
                    node,
                    "Boolean mask index_put only supports scalar values.",
                )
                return False

            if len(node.args) > 3 and node.args[3]:
                self.reporter.report_reject(
                    node,
                    "Bool-mask index_put not supported with accumulate = True.",
                )
                return False

        return True
