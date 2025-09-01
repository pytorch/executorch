# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    targets = [
        exir_ops.edge.aten._to_copy.default,
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
        merged_dtypes = copy.deepcopy(
            dtypes1
        )  # Use deepcopy to avoid unintentionally modifying SUPPORTED_INT_TYPES
        for k, v in dtypes2.items():
            merged_dtypes[k] = merged_dtypes.get(k, []) + v
        return merged_dtypes

    SUPPORTED_INT_TYPES: SupportedTypeDict = {
        torch.bool: [torch.int8, torch.int16, torch.int32],
        torch.int8: [torch.bool, torch.int16, torch.int32],
        torch.int16: [torch.bool, torch.int8, torch.int32],
        torch.int32: [torch.bool, torch.int8, torch.int16],
        torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32],
    }
    SUPPORTED_FLOAT_TYPES: SupportedTypeDict = {
        torch.int8: [torch.float16, torch.bfloat16, torch.float32],
        torch.int16: [torch.float16, torch.bfloat16, torch.float32],
        torch.int32: [torch.float16, torch.bfloat16, torch.float32],
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
        torch.bfloat16: [torch.int8, torch.int16, torch.int32, torch.float32],
        torch.float16: [torch.int8, torch.int16, torch.int32, torch.float32],
        torch.float32: [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.bfloat16,
            torch.float16,
        ],
    }
    ALL_SUPPORTED_TYPES = _merge_supported_types(
        SUPPORTED_INT_TYPES, SUPPORTED_FLOAT_TYPES
    )

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:

        supported_dtypes: SupportedTypeDict = {}
        if tosa_spec.support_integer():
            supported_dtypes = self._merge_supported_types(
                self.SUPPORTED_INT_TYPES, supported_dtypes
            )
        if tosa_spec.support_float():
            supported_dtypes = self._merge_supported_types(
                self.SUPPORTED_FLOAT_TYPES, supported_dtypes
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

        # Check memory format (to_copy)
        if "memory_format" in node.kwargs:
            if node.kwargs["memory_format"] in (torch.preserve_format,):
                self.reporter.report_reject(
                    node,
                    (
                        "Argument 'memory_format' is not supported for "
                        f"{node.target} right now."
                    ),
                )
                return False

        # Check dim_order (to_dim_order_copy)
        if "dim_order" in node.kwargs:
            dim_order = node.kwargs["dim_order"]
            # pyre-ignore[6]
            if dim_order is not None and dim_order != list(range(len(dim_order))):  # type: ignore[arg-type]
                self.reporter.report_reject(
                    node,
                    (
                        f"Argument {dim_order=} is not supported for "
                        f"{node.target} right now."
                    ),
                )
                return False

        return True
