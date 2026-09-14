# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA support checks for ``aten.index.Tensor``.

Reject unsupported indexing layouts, zero-sized tensors, and cases that exceed
``int32`` element limits.

"""

import math
from typing import cast, Sequence

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


def _has_leading_full_slices_only(indices) -> bool:
    found_tensor_index = False
    for index in indices:
        if index is None and found_tensor_index:
            return False
        found_tensor_index |= index is not None
    return found_tensor_index


@register_tosa_support_check
class IndexTensorSupported(SupportedTOSAOperatorCheck):
    """Prevent partitioning of unsupported ``index.Tensor`` usages.

    This support check is intended to prevent the partitioning of
    currently unsupported usages of the index.Tensor operator.

    1. Usages where a slice, ellipsis, or None separates indexing tensors:
        t[indexTensor, {start}:{end}, indexTensor] - slicing
        t[indexTensor, None, indexTensor] - unsqueeze
        t[indexTensor, ..., indexTensor] - ellipsis

    2. Usages where the value tensor contains more than int32.max elements
        This is due to int32 TOSA limitation and the fact that we flatten out
        and accumulate all index tensors.
        As such to avoid overflow we reject lowering of this operator if it is
        possible for indices to go over the int32 limit.

    3. Usages where the value or an index tensor is zero-sized, because TOSA
        requires every tensor dimension to be at least one.

    Extra information regarding #1:
        Pytorch decomposes slice and None usages before they reach aten.
        In the case of Slicing and Unsqueeze, Pytorch will add the relevant
        operation just before the index.Tensor op.
        In the case of Ellipsis no extra operation is added.

        The purpose of None is to signify to index.Tensor that a dimension
        should not be indexed.
        A leading run of None entries behaves like batching along those
        dimensions and is supported. None entries after the first tensor index
        remain unsupported because they interleave preserved and indexed
        dimensions.

    Examples:
        #1 - Slice -----------------------------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[1:5, torch.arange(3)]

        Turns into: (edge pseudo code)
        slice_res = ...edge__ops_aten_slice_copy_Tensor(t, dim=0, start=1, end=2)
        out = ...edge__ops_aten_index_Tensor(slice_res, [None, torch.arange(3)])

        #2 - None (Unsqueeze) ------------------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[None, torch.arange(3)]

        Turns into: edge pseudo code)
        unsqueeze_res = ...edge__ops_aten_unsqueeze(t, dim=0)
        out = ...edge__ops_aten_index_Tensor(unsqueeze_res, [None, torch.arange(3)])

        #3 - None (Unsqueeze) After index ------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[torch.arange(3), None]

        Turns into: edge pseudo code)
        unsqueeze_res = ...edge__ops_aten_unsqueeze(t, dim=1)
        out = ...edge__ops_aten_index_Tensor(unsqueeze_res, [torch.arange(3)])

    NB.
        Note that slice ops interleaved between indexes such as:
            t[1:3, torch.arange(5), 2:3, torch.arange(3).reshape(3,1)]
        are also possible and can result in some unintuitive behaviors
        where batching and indexing are mixed together.

    """

    targets = [exir_ops.edge.aten.index.Tensor]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        """Return True if ``aten.index.Tensor`` usage fits supported patterns.

        Enforces the following constraints:
        - ``None`` entries may only form a leading run before all tensor indices.
        - At least one tensor index is present.
        - Value and index tensors must not be zero-sized.
        - Boolean and byte mask indices are not supported.
        - The value tensor element count fits in ``int32``.

        """
        indices = cast(Sequence[fx.Node | None], node.args[1])
        if not _has_leading_full_slices_only(indices):
            self.reporter.report_reject(
                node,
                "Only leading None entries followed by tensor indices are supported.",
            )
            return False

        if any(
            get_first_fake_tensor(ensure_type(fx.Node, index)).dtype
            in (torch.bool, torch.uint8)
            for index in indices
            if index is not None
        ):
            self.reporter.report_reject(
                node, "Boolean and byte mask indices are not supported."
            )
            return False

        input_node = ensure_type(torch.fx.Node, node.args[0])
        input_val = get_first_fake_tensor(input_node)
        total_vals = math.prod(input_val.shape)
        has_zero_sized_index = any(
            math.prod(get_first_fake_tensor(ensure_type(fx.Node, index)).shape) == 0
            for index in indices
            if index is not None
        )
        if total_vals == 0 or has_zero_sized_index:
            self.reporter.report_reject(
                node,
                "Zero-sized value or index tensors are not supported by TOSA.",
            )
            return False

        if total_vals > torch.iinfo(torch.int32).max:
            self.reporter.report_reject(
                node,
                ("Value size exceeds int32 range; would overflow flattened indexing."),
            )
            return False

        values_dtype = input_val.dtype
        if values_dtype in (torch.bool, torch.int8, torch.int16, torch.int32):
            if not tosa_spec.support_integer():
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires INT profile.",
                )
                return False
        elif values_dtype in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            if values_dtype == torch.bfloat16 and not tosa_spec.support_extension(
                "bf16"
            ):
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires bf16 extension.",
                )
                return False
            if values_dtype == torch.float8_e4m3fn and not tosa_spec.support_extension(
                "fp8e4m3"
            ):
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires fp8e4m3 extension.",
                )
                return False
            if values_dtype == torch.float8_e5m2 and not tosa_spec.support_extension(
                "fp8e5m2"
            ):
                self.reporter.report_reject(
                    node,
                    f"{node.target}: dtype {values_dtype} requires fp8e5m2 extension.",
                )
                return False
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
                "expected bool/int8/int16/int32/float16/bfloat16/float32/float8_e4m3fn/float8_e5m2.",
            )
            return False

        return True
