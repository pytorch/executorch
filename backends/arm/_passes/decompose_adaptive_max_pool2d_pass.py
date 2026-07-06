# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.max_pool2d import (
    compute_max_pool2d_output_shape,
)
from executorch.backends.arm.tosa.specification import get_context_shape_env
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata


class DecomposeAdaptiveMaxPool2dPass(ArmPass):
    """Decompose irregular TOSA MAX_POOL2D_ADAPTIVE into per-bin slices.

    For dynamic-shape cases where ``MAX_POOL2D_ADAPTIVE`` cannot directly map
    pooling regions (input_size % output_size not in {0, 1}), materialize
    adaptive bins via ``tosa.SLICE`` and pool each bin to 1x1 with
    ``MAX_POOL2D_ADAPTIVE``.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def _is_static_dim(dim) -> bool:
        return not isinstance(dim, torch.SymInt)

    def _symbolic_bin_bounds(self, input_size, output_size: int, out_idx: int, meta):
        # Compute symbolic slice bounds directly via Python arithmetic
        start = (input_size * out_idx) // output_size
        end = (input_size * (out_idx + 1) + (output_size - 1)) // output_size
        size = end - start
        return start, size

    def _emit_tosa_slice(self, x, start_h, size_h, start_w, size_w, meta):
        # Slice the transposed NHWC tensor along its spatial axes.
        batch = x.data.shape[0]
        channel = x.data.shape[3]
        start = [0, start_h, start_w, 0]
        size = [batch, size_h, size_w, channel]
        return super().call_operator(
            exir_ops.backend.tosa.SLICE.default,
            (x, start, size),
            {},
            meta,
            True,
        )

    def _emit_adaptive_max_pool(self, x_slice, size_h, size_w, meta):
        # Use direct lists for kernel, stride, and pad
        kernel = [size_h, size_w]
        stride = [1, 1]
        pad = [0, 0, 0, 0]
        pad = super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default,
            (pad,),
            {},
            meta,
        )
        kernel = [size_h, size_w]
        if all(isinstance(k, int) for k in kernel):
            kernel = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (kernel,),
                {},
                meta,
            )
        if all(isinstance(s, int) for s in stride):
            stride = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (stride,),
                {},
                meta,
            )
        return super().call_operator(
            exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default,
            (x_slice, kernel, stride, pad),
            {},
            meta,
            True,
        )

    def _is_directly_representable(self, input_size, output_size) -> bool:
        if isinstance(output_size, torch.SymInt):
            return False
        if self._is_static_dim(input_size):
            return input_size % output_size in (0, 1)

        try:
            remainder_range = get_context_shape_env().bound_sympy(
                (input_size % output_size).node.expr
            )
        except Exception:
            return False
        return remainder_range.is_singleton() and remainder_range.upper in (0, 1)

    def _decompose_irregular(self, x, output_size_h: int, output_size_w: int, meta):
        metadata_dict = dict(meta.data)
        metadata_dict["input_qparams"] = {}
        metadata_dict["output_qparams"] = {}
        meta_with_no_qparams = NodeMetadata(metadata_dict)

        x_nhwc = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x, list(NHWC_ORDER)),
            {},
            meta,
            True,
        )
        input_h_shape = x_nhwc.data.shape[1]
        input_w_shape = x_nhwc.data.shape[2]

        rows = []
        for out_i in range(output_size_h):
            cols = []
            start_h, size_h = self._symbolic_bin_bounds(
                input_h_shape, output_size_h, out_i, meta_with_no_qparams
            )
            for out_j in range(output_size_w):
                start_w, size_w = self._symbolic_bin_bounds(
                    input_w_shape, output_size_w, out_j, meta_with_no_qparams
                )
                x_slice = self._emit_tosa_slice(
                    x_nhwc, start_h, size_h, start_w, size_w, meta_with_no_qparams
                )
                cols.append(
                    self._emit_adaptive_max_pool(
                        x_slice, size_h, size_w, meta_with_no_qparams
                    )
                )

            rows.append(
                super().call_operator(
                    exir_ops.edge.aten.cat.default,
                    (cols, 2),
                    {},
                    meta_with_no_qparams,
                    True,
                )
                if len(cols) > 1
                else cols[0]
            )

        out_nhwc = (
            super().call_operator(
                exir_ops.edge.aten.cat.default,
                (rows, 1),
                {},
                meta_with_no_qparams,
                True,
            )
            if len(rows) > 1
            else rows[0]
        )
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (out_nhwc, list(NHWC_INVERSE_ORDER)),
            {},
            meta,
            True,
        )

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op != exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default:
            return super().call_operator(op, args, kwargs, meta, updated)

        x, kernel, stride, pad = args
        output_shape = compute_max_pool2d_output_shape(
            x.data.permute(0, 2, 3, 1),
            kernel,
            stride,
            pad,
            op="MAX_POOL2D_ADAPTIVE",
        )
        output_size_h = output_shape[1]
        output_size_w = output_shape[2]

        if isinstance(output_size_h, torch.SymInt) or isinstance(
            output_size_w, torch.SymInt
        ):
            return super().call_operator(op, args, kwargs, meta, updated)

        if output_size_h <= 1 and output_size_w <= 1:
            return super().call_operator(op, args, kwargs, meta, updated)

        input_size_h, input_size_w = x.data.shape[2], x.data.shape[3]
        # If both spatial dimensions satisfy the direct-representability criterion
        # (input_size % output_size is 0 or 1 for static sizes, or symbolically
        # guaranteed in [0,1]), we can invoke the TOSA MAX_POOL2D_ADAPTIVE operator
        # directly instead of decomposing into individual bins.
        if self._is_directly_representable(
            input_size_h, output_size_h
        ) and self._is_directly_representable(input_size_w, output_size_w):
            return super().call_operator(op, args, kwargs, meta, updated)

        return self._decompose_irregular(x, output_size_h, output_size_w, meta)
