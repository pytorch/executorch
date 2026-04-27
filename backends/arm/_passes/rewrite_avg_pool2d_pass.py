# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from .fuse_constant_ops_pass import ComputeConstantOpsAOTPass


class RewriteAvgPool2dPass(ArmPass):
    """Rewrite aten.avg_pool2d calls to TOSA AVG_POOL2D op."""

    # Target the original avg_pool2d operator
    targeted_ops = {exir_ops.edge.aten.avg_pool2d.default}
    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOTPass,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):

        # Only rewrite avg_pool2d
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        x = args[0]
        pad_h, pad_w = args[3]
        # Make sure pad corresponds to TOSA
        pad = [pad_h, pad_w, pad_h, pad_w]

        _, _, h, w = x.data.shape
        kernel_h, kernel_w = args[1]
        stride_h, stride_w = args[2]

        ceil_mode = args[4] if len(args) > 4 else False

        # Adjust padding if necessary
        pad[1] = adjust_pooling_pad_if_needed(h, kernel_h, stride_h, pad[1], ceil_mode)
        pad[3] = adjust_pooling_pad_if_needed(w, kernel_w, stride_w, pad[3], ceil_mode)

        # Materialize zero-point constants
        in_qparams = meta.data.get("input_qparams", {})
        in_zp_val = in_qparams[0].get_zp_per_tensor() if 0 in in_qparams else 0
        # Materialize input zero-point as a scalar tensor
        input_zp = super().call_scalar(in_zp_val, meta)

        out_qparams = meta.data.get("output_qparams", {})
        out_zp_val = out_qparams[0].get_zp_per_tensor() if 0 in out_qparams else 0
        # Materialize output zero-point as a scalar tensor
        output_zp = super().call_scalar(out_zp_val, meta)

        # Determine accumulator dtype for AVG_POOL2D: INT32 for integer inputs, FP32 otherwise
        if x.data.dtype in (torch.int8, torch.int16):
            acc_type = torch.int32
        else:
            acc_type = torch.float32

        tosa_args = (args[0], input_zp, output_zp, *args[1:3], pad, acc_type)

        # Emit TOSA AVG_POOL2D with normalized args
        return super().call_operator(
            exir_ops.backend.tosa.AVG_POOL2D.default,
            tosa_args,
            {},
            meta,
            True,
        )
