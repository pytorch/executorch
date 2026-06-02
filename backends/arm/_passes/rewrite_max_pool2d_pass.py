# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import to_2tuple
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_max_pool2d_ops = (exir_ops.edge.aten.max_pool2d.default,)


class RewriteMaxPool2dPass(ArmOpTargetedPass):
    """Rewrite max_pool2d ops to TOSA MAX_POOL2D.

    Symbolic direct cases that match the TOSA adaptive mapping constraints are
    lowered to MAX_POOL2D_ADAPTIVE instead.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = edge_max_pool2d_ops

    @staticmethod
    def _supports_adaptive_pool() -> bool:
        try:
            tosa_spec = get_context_spec()
        except Exception:
            return False
        return (
            tosa_spec.version.major == 1
            and tosa_spec.version.minor >= 1
            and tosa_spec.support_extension("shape")
        )

    @staticmethod
    def _is_symbolic_dim(dim) -> bool:
        return isinstance(dim, torch.SymInt)

    @classmethod
    def _is_directly_representable(
        cls,
        input_size,
        kernel_size: int,
        stride: int,
        pre_pad: int | torch.SymInt,
        post_pad: int | torch.SymInt,
    ) -> bool:
        output_size = (input_size + pre_pad + post_pad - kernel_size) // stride + 1
        if cls._is_symbolic_dim(output_size):
            shape_env = get_context_shape_env()
            try:
                remainder_range = shape_env.bound_sympy(
                    (input_size % output_size).node.expr
                )
            except Exception:
                return False
            return remainder_range.is_singleton() and int(remainder_range.upper) in (
                0,
                1,
            )
        return input_size % output_size in (0, 1)

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in edge_max_pool2d_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        x = args[0]
        kernel = args[1]
        stride = to_2tuple(args[2]) if len(args) > 2 else ()
        if not stride:
            stride = kernel  # default to kernel_size

        padding = to_2tuple(args[3]) if len(args) > 3 else (0, 0)
        dilation = to_2tuple(args[4]) if len(args) > 4 else (1, 1)
        ceil_mode = args[5] if len(args) > 5 else False

        if not dilation == (1, 1):
            from executorch.backends.arm._passes.decompose_maxpool2d_with_dilation_pass import (
                DecomposeMaxPool2dPass,
            )

            raise RuntimeError(
                f"Dilation > 1 is not supported for tosa.MAX_POOL2D, has {DecomposeMaxPool2dPass.__name__} run?"
            )

        h, w = x.data.shape[2], x.data.shape[3]
        dynamic_spatial_shape = self._is_symbolic_dim(h) or self._is_symbolic_dim(w)

        # TOSA MAX_POOL2D pad order is [top, bottom, left, right].
        pad = [padding[0], padding[0], padding[1], padding[1]]
        pad[1] = adjust_pooling_pad_if_needed(
            h, kernel[0], stride[0], pad[1], ceil_mode
        )
        pad[3] = adjust_pooling_pad_if_needed(
            w, kernel[1], stride[1], pad[3], ceil_mode
        )

        # MAX_POOL2D_ADAPTIVE must use the adjusted trailing pad so the padded
        # extent is fully covered by the adaptive bins.
        if (
            dynamic_spatial_shape
            and not ceil_mode
            and self._supports_adaptive_pool()
            and self._is_directly_representable(h, kernel[0], stride[0], pad[0], pad[1])
            and self._is_directly_representable(w, kernel[1], stride[1], pad[2], pad[3])
        ):
            pre_permute = super().call_operator(
                exir_ops.edge.aten.permute_copy.default,
                (x, list(NHWC_ORDER)),
                {},
                meta,
                updated=True,
            )
            if all(isinstance(k, int) for k in kernel):
                kernel = super().call_shape_operator(
                    exir_ops.backend.tosa.CONST_SHAPE.default,
                    (list(kernel),),
                    {},
                    meta,
                )
            if all(isinstance(s, int) for s in stride):
                stride = super().call_shape_operator(
                    exir_ops.backend.tosa.CONST_SHAPE.default,
                    (list(stride),),
                    {},
                    meta,
                )
            if all(isinstance(p, int) for p in pad):
                pad = super().call_shape_operator(
                    exir_ops.backend.tosa.CONST_SHAPE.default,
                    (pad,),
                    {},
                    meta,
                )
            tosa_pool = super().call_operator(
                exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default,
                (pre_permute, kernel, stride, pad),
                {},
                meta,
                updated=True,
            )
            return super().call_operator(
                exir_ops.edge.aten.permute_copy.default,
                (tosa_pool, list(NHWC_INVERSE_ORDER)),
                {},
                meta,
                updated=True,
            )

        pre_permute = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x, list(NHWC_ORDER)),
            {},
            meta,
            updated=True,
        )

        tosa_pool = super().call_operator(
            exir_ops.backend.tosa.MAX_POOL2D.default,
            (
                pre_permute,
                list(kernel),
                list(stride),
                pad,
            ),
            {},
            meta,
            updated=True,
        )
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (tosa_pool, list(NHWC_INVERSE_ORDER)),
            {},
            meta,
            updated=True,
        )
