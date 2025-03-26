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
class ToCopySupported(SupportedTOSAOperatorCheck):
    targets = [
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    SupportedTypeDict = dict[torch.dtype, list[torch.dtype]]

    @staticmethod
    def _merge_supported_types(
        # pyre-ignore[11]
        dtypes1: SupportedTypeDict,
        dtypes2: SupportedTypeDict,
    ) -> SupportedTypeDict:
        merged_dtypes = dtypes1
        for k, v in dtypes2.items():
            merged_dtypes[k] = merged_dtypes.get(k, []) + v
        return merged_dtypes

    SUPPORTED_INT_TYPES: SupportedTypeDict = {
        torch.bool: [torch.int8, torch.int16, torch.int32],
        torch.int8: [torch.bool, torch.int16, torch.int32],
        torch.int16: [torch.bool, torch.int8, torch.int32],
        torch.int32: [torch.bool, torch.int8, torch.int16],
    }
    SUPPORTED_FLOAT_TYPES: SupportedTypeDict = {
        torch.int8: [torch.float16, torch.bfloat16, torch.float32],
        torch.int16: [torch.float16, torch.bfloat16, torch.float32],
        torch.int32: [torch.float16, torch.bfloat16, torch.float32],
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
    POSSIBLE_TYPE_CONVERSIONS = {torch.int64: torch.int32}

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        assert node.target in self.targets

        assert tosa_spec.support_integer()
        supported_dtypes = (
            self.ALL_SUPPORTED_TYPES
            if tosa_spec.support_float()
            else self.SUPPORTED_INT_TYPES
        )
        # Take into account possible type conversions
        supported_dtypes.update(
            (k, supported_dtypes[v])
            for k, v in self.POSSIBLE_TYPE_CONVERSIONS.items()
            if v in supported_dtypes
        )

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
        if output_val.dtype not in supported_dtypes[input_dtype]:
            self.reporter.report_reject(
                node,
                f"Output dtype {output_val.dtype} is not supported in "
                f"{node.target} for input dtype {input_dtype}. "
                f"Supported output types: "
                f"{''.join(str(t) for t in supported_dtypes[input_dtype])}",
            )
            return False

        # Check memory format (to_copy)
        if "memory_format" in node.kwargs:
            if node.kwargs["memory_format"] in (torch.preserve_format,):
                self.reporter.report_reject(
                    node,
                    f"Argument 'memory_format' is not supported for "
                    f"{node.target} right now.",
                )
                return False

        # Check dim_order (to_dim_order_copy)
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
