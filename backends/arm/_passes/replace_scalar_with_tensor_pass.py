# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Set, Type, Union

import torch
from executorch.backends.arm._passes.insert_table_ops import TableOps

from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass

from .arm_pass import ArmPass


# Operators that are included for both TOSA profiles
_common_ops: Dict[
    Union[EdgeOpOverload, torch._ops.OpOverload],
    Union[EdgeOpOverload, torch._ops.OpOverload],
] = {
    exir_ops.edge.aten.add.Scalar: exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Scalar: exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Scalar: exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.__rshift__.Scalar: exir_ops.edge.aten.bitwise_right_shift.Tensor,
    exir_ops.edge.aten.__lshift__.Scalar: exir_ops.edge.aten.bitwise_left_shift.Tensor,
    exir_ops.edge.aten.eq.Scalar: exir_ops.edge.aten.eq.Tensor,
    exir_ops.edge.aten.gt.Scalar: exir_ops.edge.aten.gt.Tensor,
    exir_ops.edge.aten.ge.Scalar: exir_ops.edge.aten.ge.Tensor,
    exir_ops.edge.aten.lt.Scalar: exir_ops.edge.aten.lt.Tensor,
    exir_ops.edge.aten.le.Scalar: exir_ops.edge.aten.le.Tensor,
    exir_ops.edge.aten.ne.Scalar: exir_ops.edge.aten.ne.Tensor,
    exir_ops.edge.aten.bitwise_and.Scalar: exir_ops.edge.aten.bitwise_and.Tensor,
    exir_ops.edge.aten.bitwise_or.Scalar: exir_ops.edge.aten.bitwise_or.Tensor,
    exir_ops.edge.aten.bitwise_xor.Scalar: exir_ops.edge.aten.bitwise_xor.Tensor,
    exir_ops.edge.aten.remainder.Scalar: exir_ops.edge.aten.remainder.Tensor,
    torch.ops.aten.add.Scalar: torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Scalar: torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Scalar: torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Scalar: torch.ops.aten.div.Tensor,
    torch.ops.aten.__rshift__.Scalar: torch.ops.aten.bitwise_right_shift.Tensor,
    torch.ops.aten.__lshift__.Scalar: torch.ops.aten.bitwise_left_shift.Tensor,
    torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
    torch.ops.aten.gt.Scalar: torch.ops.aten.gt.Tensor,
    torch.ops.aten.ge.Scalar: torch.ops.aten.ge.Tensor,
    torch.ops.aten.lt.Scalar: torch.ops.aten.lt.Tensor,
    torch.ops.aten.le.Scalar: torch.ops.aten.le.Tensor,
    torch.ops.aten.ne.Scalar: torch.ops.aten.ne.Tensor,
    torch.ops.aten.bitwise_and.Scalar: torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.bitwise_or.Scalar: torch.ops.aten.bitwise_or.Tensor,
    torch.ops.aten.bitwise_xor.Scalar: torch.ops.aten.bitwise_xor.Tensor,
    torch.ops.aten.remainder.Scalar: torch.ops.aten.remainder.Tensor,
}

_fp_profile_ops: Dict[
    Union[EdgeOpOverload, torch._ops.OpOverload],
    Union[EdgeOpOverload, torch._ops.OpOverload],
] = _common_ops | {
    exir_ops.edge.aten.pow.Tensor_Scalar: exir_ops.edge.aten.pow.Tensor_Tensor,
}

_int_profile_ops: Dict[
    Union[EdgeOpOverload, torch._ops.OpOverload],
    Union[EdgeOpOverload, torch._ops.OpOverload],
] = _common_ops

_all_ops: Dict[
    Union[EdgeOpOverload, torch._ops.OpOverload],
    Union[EdgeOpOverload, torch._ops.OpOverload],
] = (
    _fp_profile_ops | _int_profile_ops
)


class ReplaceScalarWithTensorByProfilePass(ArmPass, ReplaceScalarWithTensorArgPass):
    """Profile-aware scalar-to-tensor replacement pass for binary ops."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, tfa_pass=False, *args, **kwargs):
        # NOTE diamond heritance for this class, thus MRO is important.

        # Initialize base (ReplaceScalarWithTensorArgPass) with the full
        # superset which will make the superclass handle ops in _all_ops.
        # Actual selection is done per-call in call_operator.
        super().__init__(tfa_pass, _all_ops, *args, **kwargs)

    def call_operator(self, op, args, kwargs, meta):
        tosa_spec = get_context_spec()

        included_ops = {}
        if tosa_spec.support_integer():
            included_ops |= _int_profile_ops
        if tosa_spec.support_float():
            included_ops |= _fp_profile_ops

        if included_ops == {}:
            raise ValueError("Profile must support at least INT or FP")

        if op in TableOps.included_ops():
            # Do not handle quantized table ops; forward unchanged.
            input_qparams = meta.data.get("input_qparams", {})
            output_qparams = meta.data.get("input_qparams", {})
            if len(input_qparams) > 0 and len(output_qparams) > 0:
                # Do not handle; forward unchanged.
                return ExportPass.call_operator(self, op, args, kwargs, meta)

        if op in included_ops:
            # Include this op based on the current profile.
            return super().call_operator(op, args, kwargs, meta)
        else:
            # Do not handle; forward unchanged.
            return ExportPass.call_operator(self, op, args, kwargs, meta)
