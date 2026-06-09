# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass

from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewriteAdaptiveAvgPool2dPass(ArmPass):
    """Rewrite dynamic adaptive average pooling to tosa.avg_pool2d_adaptive when
    possible.

    The condition for rewriting is that symbolic input dimensions have a known
    remainder of 0 or 1 when divided by the static output dimensions. This
    preserves the adaptive pooling regions without materializing slice/cat
    decomposition.

    """

    targeted_ops = {exir_ops.edge.aten._adaptive_avg_pool2d.default}
    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOTPass,
    }

    @staticmethod
    def _is_symbolic_dim(dim) -> bool:
        return isinstance(dim, torch.SymInt)

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

    @classmethod
    def _get_pool_params(cls, input_size, output_size: int):
        if isinstance(output_size, torch.SymInt) or not isinstance(output_size, int):
            return None

        remainder = input_size % output_size
        if cls._is_symbolic_dim(remainder):
            shape_env = get_context_shape_env()
            try:
                remainder_range = shape_env.bound_sympy(remainder.node.expr)
            except Exception:
                return None

            if not remainder_range.is_singleton() or int(remainder_range.upper) not in (
                0,
                1,
            ):
                return None

            stride = input_size // output_size
            return stride + int(remainder_range.upper), stride

        if remainder not in (0, 1):
            return None

        stride = input_size // output_size
        return stride + remainder, stride

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        x = args[0]
        _, _, input_h, input_w = x.data.shape
        if not (self._is_symbolic_dim(input_h) or self._is_symbolic_dim(input_w)):
            return super().call_operator(op, args, kwargs, meta, updated)

        # Dynamic adaptive lowering requires shape-aware TOSA support.
        if not self._supports_dynamic_tosa_adaptive():
            raise RuntimeError(
                "Dynamic adaptive_avg_pool2d rewrite requires TOSA-1.1 with the shape extension."
            )

        output_h, output_w = args[1]
        h_params = self._get_pool_params(input_h, output_h)
        w_params = self._get_pool_params(input_w, output_w)
        # Fall back when either spatial dimension cannot be expressed as one TOSA adaptive pool.
        if h_params is None or w_params is None:
            return super().call_operator(op, args, kwargs, meta, updated)

        kernel = [h_params[0], w_params[0]]
        stride = [h_params[1], w_params[1]]
        pad = [0, 0, 0, 0]
        pad = super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default,
            (pad,),
            {},
            meta,
        )
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

        in_qparams = meta.data.get("input_qparams", {})
        in_zp_val = in_qparams[0].get_zp_per_tensor() if 0 in in_qparams else 0
        input_zp = self.call_scalar(in_zp_val, meta)

        out_qparams = meta.data.get("output_qparams", {})
        out_zp_val = out_qparams[0].get_zp_per_tensor() if 0 in out_qparams else 0
        output_zp = self.call_scalar(out_zp_val, meta)

        acc_type = (
            torch.int32 if x.data.dtype in (torch.int8, torch.int16) else torch.float32
        )
        pre_permute = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x, list(NHWC_ORDER)),
            {},
            meta,
            True,
        )
        tosa_args = (
            pre_permute,
            input_zp,
            output_zp,
            kernel,
            stride,
            pad,
            acc_type,
        )

        tosa_avg_pool = super().call_operator(
            exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default,
            tosa_args,
            {},
            meta,
            True,
        )
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (tosa_avg_pool, list(NHWC_INVERSE_ORDER)),
            {},
            meta,
            True,
        )
