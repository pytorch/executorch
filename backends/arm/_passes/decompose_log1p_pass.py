# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.match_arg_dtype_pass import MatchArgDtypePass
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeLog1pPass(ArmPass):
    """Decompose log1p into a small polynomial with a log fallback for larger inputs."""

    _passes_required_after: Set[Type[ExportPass]] = {
        InsertTableOpsPass,
        MatchArgRanksPass,
        MatchArgDtypePass,
        ReplaceScalarWithTensorByProfilePass,
    }

    _supported_ops = {
        exir_ops.edge.aten.log1p.default,
    }

    def _poly(self, x, meta):
        # 6-term Taylor: x - x^2/2 + x^3/3 - x^4/4 + x^5/5 - x^6/6
        op_mul = exir_ops.edge.aten.mul.Tensor
        op_mul_scalar = exir_ops.edge.aten.mul.Scalar
        op_add = exir_ops.edge.aten.add.Tensor

        x2 = super().call_operator(op_mul, (x, x), {}, meta, updated=True)
        x3 = super().call_operator(op_mul, (x2, x), {}, meta, updated=True)
        x4 = super().call_operator(op_mul, (x3, x), {}, meta, updated=True)
        x5 = super().call_operator(op_mul, (x4, x), {}, meta, updated=True)
        x6 = super().call_operator(op_mul, (x5, x), {}, meta, updated=True)

        t2 = super().call_operator(op_mul_scalar, (x2, -0.5), {}, meta, updated=True)
        t3 = super().call_operator(
            op_mul_scalar, (x3, 1.0 / 3.0), {}, meta, updated=True
        )
        t4 = super().call_operator(op_mul_scalar, (x4, -0.25), {}, meta, updated=True)
        t5 = super().call_operator(op_mul_scalar, (x5, 0.2), {}, meta, updated=True)
        t6 = super().call_operator(
            op_mul_scalar, (x6, -1.0 / 6.0), {}, meta, updated=True
        )

        acc = super().call_operator(op_add, (x, t2), {}, meta, updated=True)
        acc = super().call_operator(op_add, (acc, t3), {}, meta, updated=True)
        acc = super().call_operator(op_add, (acc, t4), {}, meta, updated=True)
        acc = super().call_operator(op_add, (acc, t5), {}, meta, updated=True)
        acc = super().call_operator(op_add, (acc, t6), {}, meta, updated=True)
        return acc

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._supported_ops:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        is_quantized = (
            len(meta.data.get("input_qparams", {})) > 0
            and len(meta.data.get("output_qparams", {})) > 0
        )
        if is_quantized:
            # Quantized log1p should be handled by LUT/table instead of decomposition.
            return super().call_operator(op, args, kwargs, meta)

        logging.info("Decomposing log1p via polynomial + log branch for FP profile.")

        x = args[0]
        approx = self._poly(x, meta)

        # For x > 1e-3, use log(1+x) directly.
        op_add_scalar = exir_ops.edge.aten.add.Scalar
        op_gt = exir_ops.edge.aten.gt.Scalar
        op_where = exir_ops.edge.aten.where.self
        op_log = exir_ops.edge.aten.log.default

        one_plus_x = super().call_operator(
            op_add_scalar, (x, 1.0), {}, meta, updated=True
        )
        log_branch = super().call_operator(
            op_log, (one_plus_x,), {}, meta, updated=True
        )

        mask = super().call_operator(op_gt, (x, 1e-3), {}, meta, updated=True)
        result = super().call_operator(
            op_where, (mask, log_branch, approx), {}, meta, updated=True
        )
        return result
