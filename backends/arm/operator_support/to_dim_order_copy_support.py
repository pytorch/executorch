# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for ``_to_dim_order_copy`` in TOSA.

Provide dtype-compatibility checks for casting when converting to a specific
dimension order. Supported input/output dtype pairs depend on the active TOSA
profile (integer and/or float).

"""

# pyre-unsafe
import copy
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

SupportedTypeDict = dict[torch.dtype, list[torch.dtype]]


@register_tosa_support_check
class ToCopySupported(SupportedTOSAOperatorCheck):
    """Provide TOSA support check for ``_to_dim_order_copy``.

    Attributes:
        SUPPORTED_INT_PROFILE_DTYPES (dict[torch.dtype, list[torch.dtype]]):
            Allowed output dtypes for each integer input dtype.
        SUPPORTED_FP_PROFILE_DTYPES (dict[torch.dtype, list[torch.dtype]]):
            Allowed output dtypes for each floating input dtype.

    """

    targets = [
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    @staticmethod
    def _merge_supported_types(
        # pyre-ignore[11]
        dtypes1: SupportedTypeDict,
        dtypes2: SupportedTypeDict,
    ) -> SupportedTypeDict:
        """Return a merged mapping of supported dtype transitions.

        Args:
            dtypes1 (dict[torch.dtype, list[torch.dtype]]): Base mapping.
            dtypes2 (dict[torch.dtype, list[torch.dtype]]): Mapping to merge in.

        Returns:
            dict[torch.dtype, list[torch.dtype]]: Combined mapping.

        """
        merged_dtypes = copy.deepcopy(
            dtypes1
        )  # Use deepcopy to avoid unintentionally modifying SUPPORTED_INT_PROFILE_DTYPES
        for k, v in dtypes2.items():
            merged_dtypes[k] = merged_dtypes.get(k, []) + v
        return merged_dtypes

    SUPPORTED_INT_PROFILE_DTYPES: SupportedTypeDict = {
        torch.bool: [torch.bool, torch.int8, torch.int16, torch.int32],
        torch.int8: [torch.bool, torch.int8, torch.int16, torch.int32],
        torch.int16: [torch.bool, torch.int8, torch.int16, torch.int32],
        torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],
        torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32],
    }
    SUPPORTED_FP_PROFILE_DTYPES: SupportedTypeDict = {
        torch.int8: [torch.int8, torch.float16, torch.bfloat16, torch.float32],
        torch.int16: [torch.int16, torch.float16, torch.bfloat16, torch.float32],
        torch.int32: [torch.int32, torch.float16, torch.bfloat16, torch.float32],
        # INT64 inputs to casts *should* be ok, since they should be rejected by
        # CheckInt64InputsAndOutputs if the cast can't be done AOT.
        torch.int64: [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ],
        torch.bfloat16: [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.bfloat16,
            torch.float32,
        ],
        torch.float16: [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float16,
            torch.float32,
        ],
        torch.float32: [
            torch.float32,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ],
    }

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Return True if the node is supported by TOSA.

        Check FakeTensor metadata, validate input dtype is supported for the
        active profile, and ensure the output dtype is allowed for the given
        input dtype.

        """
        supported_dtypes: SupportedTypeDict = {}
        if tosa_spec.support_integer():
            supported_dtypes = self._merge_supported_types(
                self.SUPPORTED_INT_PROFILE_DTYPES, supported_dtypes
            )
        if tosa_spec.support_float():
            supported_dtypes = self._merge_supported_types(
                self.SUPPORTED_FP_PROFILE_DTYPES, supported_dtypes
            )

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

        # Check input type
        input_dtype = input_val.dtype
        if input_dtype not in supported_dtypes:
            self.reporter.report_reject(
                node,
                f"Input dtype {input_val.dtype} is not supported in {node.target}.",
            )
            return False

        # Check output type
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
