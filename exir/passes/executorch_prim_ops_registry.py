# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Dict, Set, Union

# necessary to ensure the ops are registered
import torch
from executorch.exir.dialects._ops import bind_pattern_to_op, ops
from torch import SymBool, SymFloat, SymInt
from torch._ops import OpOverload
from torch.library import Library


executorch_prims_lib = Library("executorch_prim", "DEF")

_SymScalar = Union[SymBool, SymFloat, SymInt]


@bind_pattern_to_op(executorch_prims_lib, "add.Scalar(Scalar a, Scalar b) -> Scalar")
def add(a: _SymScalar, b: _SymScalar) -> _SymScalar:
    return a + b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "mul.Scalar(Scalar a, Scalar b) -> Scalar")
def mul(a: _SymScalar, b: _SymScalar) -> _SymScalar:
    return a * b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "sub.Scalar(Scalar a, Scalar b) -> Scalar")
def sub(a: _SymScalar, b: _SymScalar) -> _SymScalar:
    return a - b  # pyre-ignore


@bind_pattern_to_op(
    executorch_prims_lib, "floordiv.Scalar(Scalar a, Scalar b) -> Scalar"
)
def floordiv(a: _SymScalar, b: _SymScalar) -> _SymScalar:
    return a // b  # pyre-ignore


@bind_pattern_to_op(
    executorch_prims_lib, "truediv.Scalar(Scalar a, Scalar b) -> Scalar"
)
def truediv(a: _SymScalar, b: _SymScalar) -> _SymScalar:
    return a / b  # pyre-ignore


# TODO: ideally we should return SymBool in the schema, but it seems
# the schema parser does not recognize SymBool yet: P629748075
@bind_pattern_to_op(executorch_prims_lib, "gt.Scalar(Scalar a, Scalar b) -> bool")
def gt(a: _SymScalar, b: _SymScalar) -> bool:
    return a > b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "lt.Scalar(Scalar a, Scalar b) -> bool")
def lt(a: _SymScalar, b: _SymScalar) -> bool:
    return a < b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "ge.Scalar(Scalar a, Scalar b) -> bool")
def ge(a: _SymScalar, b: _SymScalar) -> bool:
    return a >= b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "le.Scalar(Scalar a, Scalar b) -> bool")
def le(a: _SymScalar, b: _SymScalar) -> bool:
    return a <= b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "eq.Scalar(Scalar a, Scalar b) -> bool")
def eq(a: _SymScalar, b: _SymScalar) -> bool:
    return a == b


_PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS: Dict[OpOverload, OpOverload] = {
    operator.sub: ops.backend.executorch_prim.sub.Scalar,
    operator.mul: ops.backend.executorch_prim.mul.Scalar,
    operator.add: ops.backend.executorch_prim.add.Scalar,
    operator.floordiv: ops.backend.executorch_prim.floordiv.Scalar,
    operator.truediv: ops.backend.executorch_prim.truediv.Scalar,
    operator.eq: ops.backend.executorch_prim.eq.Scalar,
    operator.gt: ops.backend.executorch_prim.gt.Scalar,
    operator.lt: ops.backend.executorch_prim.lt.Scalar,
    operator.ge: ops.backend.executorch_prim.ge.Scalar,
    operator.le: ops.backend.executorch_prim.le.Scalar,
}


_EXECUTORCH_SYM_OPS: Set[OpOverload] = set(
    _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.values()
)
_EXECUTORCH_SYM_OPS.update(
    {
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_numel.default,
    }
)


def is_sym_op(target) -> bool:
    return (
        target in _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.keys()
        or target in _EXECUTORCH_SYM_OPS
    )
