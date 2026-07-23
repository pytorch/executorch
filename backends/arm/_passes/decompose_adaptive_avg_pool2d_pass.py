# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, floor
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.decompose_avg_pool2d_pass import (
    DecomposeAvgPool2dPass,
)
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata

edge_ops = (exir_ops.edge.aten._adaptive_avg_pool2d.default,)
aten_ops = (torch.ops.aten.adaptive_avg_pool2d.default,)


def _get_decomposition(op) -> tuple:
    if op in edge_ops:
        return (
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.cat.default,
        )
    if op in aten_ops:
        return (
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.slice_copy.Tensor,
            torch.ops.aten.cat.default,
        )
    raise RuntimeError(f"Unable to get decomposition for op {op}")


class DecomposeAdaptiveAvgPool2dPass(ArmOpTargetedPass):
    """Decompose static-shape topology-changing AdaptiveAvgPool2d cases.

    Static input/output shapes use the existing slice + avg_pool2d + cat
    lowering, with a fast path for directly representable uniform regions.

    Dynamic cases are left untouched for dedicated dynamic rewrite/decompose
    passes later in the TOSA pipeline.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeAvgPool2dPass}
    target_ops = edge_ops + aten_ops
    check_allowed_to_transform = True

    targeted_ops = {*edge_ops, *aten_ops}

    @staticmethod
    def _is_static_dim(dim) -> bool:
        return not isinstance(dim, torch.SymInt)

    @classmethod
    def _is_static_shape(cls, *dims) -> bool:
        return all(cls._is_static_dim(dim) for dim in dims)

    @staticmethod
    def _has_dynamic_spatial_shape(x) -> bool:
        _, _, input_size_h, input_size_w = x.data.shape
        return isinstance(input_size_h, torch.SymInt) or isinstance(
            input_size_w, torch.SymInt
        )

    def _call_const_shape(self, value: int, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default,
            ([value],),
            {},
            meta,
            True,
        )

    def _get_dim_shape(self, x, axis: int, meta: NodeMetadata):
        dim = x.data.shape[axis]
        if isinstance(dim, torch.SymInt):
            return super().call_shape_operator(
                exir_ops.backend.tosa.DIM.default,
                (x,),
                {"axis": axis},
                meta,
                True,
            )
        return self._call_const_shape(dim, meta)

    def _shape_mul_const(self, value, factor: int, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.MUL_SHAPE.default,
            (value, self._call_const_shape(factor, meta)),
            {},
            meta,
            True,
        )

    def _shape_add_const(self, value, addend: int, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.ADD_SHAPE.default,
            (value, self._call_const_shape(addend, meta)),
            {},
            meta,
            True,
        )

    def _shape_floor_div_const(self, value, divisor: int, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default,
            (value, self._call_const_shape(divisor, meta)),
            {},
            meta,
            True,
        )

    def _shape_sub(self, lhs, rhs, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.SUB_SHAPE.default,
            (lhs, rhs),
            {},
            meta,
            True,
        )

    def _shape_concat(self, parts: list, meta: NodeMetadata):
        return super().call_shape_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default,
            (parts,),
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

    def _is_dynamic_direct_case(self, x, output_size_h, output_size_w) -> bool:
        _, _, input_size_h, input_size_w = x.data.shape
        if not self._has_dynamic_spatial_shape(x):
            return False
        return self._is_directly_representable(
            input_size_h, output_size_h
        ) and self._is_directly_representable(input_size_w, output_size_w)

    @staticmethod
    def _static_bin_bounds(
        input_size: int, output_size: int, out_idx: int
    ) -> tuple[int, int]:
        start = floor(out_idx * input_size / output_size)
        end = ceil((out_idx + 1) * input_size / output_size)
        return start, end

    def _symbolic_bin_bounds(self, input_size, output_size: int, out_idx: int, meta):
        start_num = self._shape_mul_const(input_size, out_idx, meta)
        start = self._shape_floor_div_const(start_num, output_size, meta)

        end_num = self._shape_mul_const(input_size, out_idx + 1, meta)
        end_num = self._shape_add_const(end_num, output_size - 1, meta)
        end = self._shape_floor_div_const(end_num, output_size, meta)

        size = self._shape_sub(end, start, meta)
        return start, end, size

    def _emit_tosa_slice(self, x, start_h, size_h, start_w, size_w, meta):
        start = self._shape_concat(
            [
                self._call_const_shape(0, meta),
                self._call_const_shape(0, meta),
                start_h,
                start_w,
            ],
            meta,
        )
        size = self._shape_concat(
            [
                self._get_dim_shape(x, 0, meta),
                self._get_dim_shape(x, 1, meta),
                size_h,
                size_w,
            ],
            meta,
        )
        return super().call_operator(
            exir_ops.backend.tosa.SLICE.default,
            (x, start, size),
            {},
            meta,
            True,
        )

    def _emit_adaptive_pool(self, x_slice, size_h, size_w, meta):
        in_qparams = meta.data.get("input_qparams", {})
        in_zp_val = in_qparams[0].get_zp_per_tensor() if 0 in in_qparams else 0
        input_zp = self.call_scalar(in_zp_val, meta)

        out_qparams = meta.data.get("output_qparams", {})
        out_zp_val = out_qparams[0].get_zp_per_tensor() if 0 in out_qparams else 0
        output_zp = self.call_scalar(out_zp_val, meta)

        acc_type = (
            torch.int32
            if x_slice.data.dtype in (torch.int8, torch.int16)
            else torch.float32
        )
        stride = [1, 1]
        pad = [0, 0, 0, 0]
        x_slice_nhwc = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x_slice, list(NHWC_ORDER)),
            {},
            meta,
            True,
        )
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
        else:
            kernel = self._shape_concat(
                [
                    self._get_dim_shape(x_slice_nhwc, 1, meta),
                    self._get_dim_shape(x_slice_nhwc, 2, meta),
                ],
                meta,
            )
        if all(isinstance(s, int) for s in stride):
            stride = super().call_shape_operator(
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (stride,),
                {},
                meta,
            )
        pooled_nhwc = super().call_operator(
            exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default,
            (x_slice_nhwc, input_zp, output_zp, kernel, stride, pad, acc_type),
            {},
            meta,
            True,
        )
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (pooled_nhwc, list(NHWC_INVERSE_ORDER)),
            {},
            meta,
            True,
        )

    @staticmethod
    def _supports_dynamic_tosa_adaptive() -> bool:
        try:
            tosa_spec = get_context_spec()
        except Exception:
            return False
        return (
            tosa_spec.version.major == 1
            and tosa_spec.version.minor >= 1
            and tosa_spec.support_extension("shape")
        )

    def _decompose_static(
        self,
        avg_pool2d_op,
        slice_op,
        cat_op,
        x,
        output_size_h,
        output_size_w,
        kwargs,
        meta,
    ):
        _, _, input_size_h, input_size_w = x.data.shape

        stride_h = floor(input_size_h / output_size_h)
        stride_w = floor(input_size_w / output_size_w)
        if (
            self._is_directly_representable(input_size_h, output_size_h)
            and self._is_directly_representable(input_size_w, output_size_w)
            and stride_h in (1, 2, 3)
            and stride_w in (1, 2, 3)
        ):
            kernel_h = stride_h + (input_size_h % output_size_h)
            kernel_w = stride_w + (input_size_w % output_size_w)
            return super().call_operator(
                avg_pool2d_op,
                (x, (kernel_h, kernel_w), (stride_h, stride_w), (0, 0)),
                kwargs,
                meta,
                True,
            )

        metadata_dict = dict(meta.data)
        metadata_dict["input_qparams"] = {}
        metadata_dict["output_qparams"] = {}
        meta_with_no_qparams = NodeMetadata(metadata_dict)

        res = []
        for out_i in range(output_size_h):
            row = []
            for out_j in range(output_size_w):
                start_h, end_h = self._static_bin_bounds(
                    input_size_h, output_size_h, out_i
                )
                start_w, end_w = self._static_bin_bounds(
                    input_size_w, output_size_w, out_j
                )

                x_h = super().call_operator(
                    slice_op,
                    (x, 2, start_h, end_h),
                    kwargs,
                    meta_with_no_qparams,
                    True,
                )
                x_hw = super().call_operator(
                    slice_op,
                    (x_h, 3, start_w, end_w),
                    kwargs,
                    meta_with_no_qparams,
                    True,
                )

                kernel_h = end_h - start_h
                kernel_w = end_w - start_w
                pooled = super().call_operator(
                    avg_pool2d_op,
                    (x_hw, (kernel_h, kernel_w), (1, 1), (0, 0)),
                    kwargs,
                    meta,
                    True,
                )
                row.append(pooled)

            row_tensor = (
                super().call_operator(
                    cat_op,
                    (row, 3),
                    kwargs,
                    meta_with_no_qparams,
                    True,
                )
                if len(row) > 1
                else row[0]
            )
            res.append(row_tensor)

        return (
            super().call_operator(
                cat_op,
                (res, 2),
                kwargs,
                meta_with_no_qparams,
                True,
            )
            if len(res) > 1
            else res[0]
        )

    def _decompose_dynamic_static_output(
        self, x, cat_op, output_size_h: int, output_size_w: int, kwargs, meta
    ):
        metadata_dict = dict(meta.data)
        metadata_dict["input_qparams"] = {}
        metadata_dict["output_qparams"] = {}
        meta_with_no_qparams = NodeMetadata(metadata_dict)

        input_h_shape = self._get_dim_shape(x, 2, meta_with_no_qparams)
        input_w_shape = self._get_dim_shape(x, 3, meta_with_no_qparams)

        res = []
        for out_i in range(output_size_h):
            row = []
            start_h, _end_h, size_h = self._symbolic_bin_bounds(
                input_h_shape, output_size_h, out_i, meta_with_no_qparams
            )
            for out_j in range(output_size_w):
                start_w, _end_w, size_w = self._symbolic_bin_bounds(
                    input_w_shape, output_size_w, out_j, meta_with_no_qparams
                )
                x_slice = self._emit_tosa_slice(
                    x, start_h, size_h, start_w, size_w, meta_with_no_qparams
                )
                pooled = self._emit_adaptive_pool(x_slice, size_h, size_w, meta)
                row.append(pooled)

            row_tensor = (
                super().call_operator(
                    cat_op,
                    (row, 3),
                    kwargs,
                    meta_with_no_qparams,
                    True,
                )
                if len(row) > 1
                else row[0]
            )
            res.append(row_tensor)

        return (
            super().call_operator(
                cat_op,
                (res, 2),
                kwargs,
                meta_with_no_qparams,
                True,
            )
            if len(res) > 1
            else res[0]
        )

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in (edge_ops + aten_ops) or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta, updated)

        avg_pool2d_op, slice_op, cat_op = _get_decomposition(op)
        x = args[0]
        output_size_h, output_size_w = args[1]

        if isinstance(output_size_h, torch.SymInt) or isinstance(
            output_size_w, torch.SymInt
        ):
            return super().call_operator(op, args, kwargs, meta, updated)

        _, _, input_size_h, input_size_w = x.data.shape
        if not self._is_static_shape(input_size_h, input_size_w):
            return super().call_operator(op, args, kwargs, meta, updated)

        return self._decompose_static(
            avg_pool2d_op,
            slice_op,
            cat_op,
            x,
            output_size_h,
            output_size_w,
            kwargs,
            meta,
        )
