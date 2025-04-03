# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Dict

import torch
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.dialects.edge._ops import EdgeOpOverload


# Operators that are included for both TOSA profiles
_common_ops: Dict[EdgeOpOverload, EdgeOpOverload] = {
    exir_ops.edge.aten.add.Scalar: exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Scalar: exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Scalar: exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.__rshift__.Scalar: exir_ops.edge.aten.bitwise_right_shift.Tensor,
    exir_ops.edge.aten.__lshift__.Scalar: exir_ops.edge.aten.bitwise_left_shift.Tensor,
    exir_ops.edge.aten.eq.Scalar: exir_ops.edge.aten.eq.Tensor,
    torch.ops.aten.add.Scalar: torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Scalar: torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Scalar: torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Scalar: torch.ops.aten.div.Tensor,
    torch.ops.aten.__rshift__.Scalar: torch.ops.aten.bitwise_right_shift.Tensor,
    torch.ops.aten.__lshift__.Scalar: torch.ops.aten.bitwise_left_shift.Tensor,
    torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
}


class ReplaceScalarWithTensorArgPassTOSAMI(ReplaceScalarWithTensorArgPass):
    scalar_to_tensor_ops = _common_ops | {
        exir_ops.edge.aten.pow.Tensor_Scalar: exir_ops.edge.aten.pow.Tensor_Tensor,
        torch.ops.aten.pow.Tensor_Scalar: torch.ops.aten.pow.Tensor_Tensor,
    }

    def __init__(self):
        super().__init__(self.scalar_to_tensor_ops)


class ReplaceScalarWithTensorArgPassTOSABI(ReplaceScalarWithTensorArgPass):
    scalar_to_tensor_ops = _common_ops

    def __init__(self):
        super().__init__(self.scalar_to_tensor_ops)
