# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declare operator support for aten.as_strided_copy in the TOSA backend."""

from collections.abc import Mapping
from typing import Any, Optional

import torch.fx as fx
from executorch.backends.arm.common.as_strided_utils import (
    contiguous_strides,
    maybe_static_sequence,
    to_int,
    to_int_tuple,
)

from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


def _arg_from_node(node: fx.Node, position: int, keyword: str) -> object | None:
    """Fetch an argument either by keyword or positional index."""
    kwargs: Mapping[str, Any] = node.kwargs
    if keyword in kwargs:
        return kwargs[keyword]
    if len(node.args) > position:
        return node.args[position]
    return None


def _extract_static_args(
    node: fx.Node,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...], int]]:
    """Return static size/stride/offset if they are compatible."""
    size_arg = _arg_from_node(node, 1, "size")
    stride_arg = _arg_from_node(node, 2, "stride")
    offset_arg = _arg_from_node(node, 3, "storage_offset")

    if (
        maybe_static_sequence(size_arg) is None
        or maybe_static_sequence(stride_arg) is None
    ):
        return None

    size_tuple = to_int_tuple(size_arg)
    stride_tuple = to_int_tuple(stride_arg)
    if size_tuple is None or stride_tuple is None:
        return None

    if len(size_tuple) != len(stride_tuple):
        return None

    if any(stride < 0 for stride in stride_tuple):
        return None

    storage_offset = 0
    if offset_arg is not None:
        parsed_offset = to_int(offset_arg)
        if parsed_offset is None:
            return None
        storage_offset = parsed_offset

    return size_tuple, stride_tuple, storage_offset


@register_tosa_support_check
class AsStridedCopySupported(SupportedTOSAOperatorCheck):
    """Support check ensuring as_strided_copy is contiguous with zero offset."""

    targets = [exir_ops.edge.aten.as_strided_copy.default]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification  # noqa: D417
    ) -> bool:
        extracted = _extract_static_args(node)
        if extracted is None:
            self.reporter.report_reject(
                node, "Size/stride must be static with non-negative strides."
            )
            return False

        size_tuple, stride_tuple, storage_offset = extracted

        if storage_offset != 0:
            self.reporter.report_reject(
                node, "Non-zero storage offsets are unsupported."
            )
            return False

        expected_strides = contiguous_strides(size_tuple)

        def _stride_matches(idx: int, dim: int) -> bool:
            stride = stride_tuple[idx]
            expected = expected_strides[idx]
            if idx == len(size_tuple) - 1:
                return stride >= expected
            if dim == 1:
                return True
            return stride == expected

        if any(not _stride_matches(i, dim) for i, dim in enumerate(size_tuple)):
            self.reporter.report_reject(
                node,
                f"Stride {stride_tuple} is not contiguous for shape {size_tuple}.",
            )
            return False

        return True
