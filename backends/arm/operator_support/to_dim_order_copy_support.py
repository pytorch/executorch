# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``_to_dim_order_copy`` in TOSA.

Provide dtype-compatibility checks for casting when converting to a specific
dimension order. Supported input/output dtype pairs depend on the active TOSA
profile (integer and/or float).

"""

import logging

import torch

import torch.fx as fx

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.cast_support import (
    supported_to_dim_order_casts,
    SupportedCastMap,
)
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


@register_tosa_support_check
class ToCopySupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``_to_dim_order_copy``."""

    targets = [
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ]

    @staticmethod
    def _is_quantized_identity_cast(node: torch.fx.Node) -> bool:
        for user in node.users:
            if (
                not user.target
                == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            ):
                return False
            scale = user.args[1]
            zp = user.args[2]
            if scale != 1.0 or zp != 0.0:
                return False
        return True

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if the node is supported by TOSA.

        Check FakeTensor metadata, validate input dtype is supported for the
        active profile, and ensure the output dtype is allowed for the given
        input dtype.

        """
        supported_dtypes: SupportedCastMap = supported_to_dim_order_casts(tosa_spec)

        if len(node.all_input_nodes) != 1:
            self.reporter.report_reject(
                node,
                (
                    "Expected exactly one input node, "
                    f"got {len(node.all_input_nodes)} for {node.target}."
                ),
            )
            return False
        input_val = node.all_input_nodes[0].meta["val"]
        if not isinstance(input_val, torch._subclasses.FakeTensor):
            self.reporter.report_reject(
                node,
                (
                    "Invalid or missing meta: expected FakeTensor input, got "
                    f"{type(input_val).__name__} for {node.target}."
                ),
            )
            return False

        input_dtype = input_val.dtype
        if input_dtype not in supported_dtypes:
            self.reporter.report_reject(
                node,
                f"Input dtype {input_val.dtype} is not supported in {node.target}.",
            )
            return False

        output_val = node.meta["val"]
        if not isinstance(output_val, torch._subclasses.FakeTensor):
            self.reporter.report_reject(
                node,
                (
                    "Invalid or missing meta: expected FakeTensor output, got "
                    f"{type(output_val).__name__} for {node.target}."
                ),
            )
            return False
        if output_val.dtype not in supported_dtypes[input_dtype]:
            if tosa_spec.support_integer() and self._is_quantized_identity_cast(node):
                return True
            self.reporter.report_reject(
                node,
                (
                    f"Output dtype {output_val.dtype} is not supported in "
                    f"{node.target} for input dtype {input_dtype}. "
                    f"Supported output types: "
                    f"{', '.join(str(t) for t in supported_dtypes[input_dtype])}"
                ),
            )
            return False

        return True
