# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Dict, Set, Type, Union

import torch
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass


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
    exir_ops.edge.aten.fmod.Scalar: exir_ops.edge.aten.fmod.Tensor,
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
    torch.ops.aten.fmod.Scalar: torch.ops.aten.fmod.Tensor,
}


class ReplaceScalarWithTensorArgPassTOSAMI(ReplaceScalarWithTensorArgPass):
    _passes_required_after: Set[Type[ExportPass]] = set()

    scalar_to_tensor_ops = _common_ops | {
        exir_ops.edge.aten.pow.Tensor_Scalar: exir_ops.edge.aten.pow.Tensor_Tensor,
        torch.ops.aten.pow.Tensor_Scalar: torch.ops.aten.pow.Tensor_Tensor,
    }

    def __init__(self):
        super().__init__(self.scalar_to_tensor_ops)


class ReplaceScalarWithTensorArgPassTOSABI(ReplaceScalarWithTensorArgPass):
    _passes_required_after: Set[Type[ExportPass]] = set()

    scalar_to_tensor_ops = _common_ops

    def __init__(self):
        super().__init__(self.scalar_to_tensor_ops)
