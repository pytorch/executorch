# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Dict, Set

# necessary to ensure the ops are registered
import torch
from executorch.exir.dialects._ops import bind_pattern_to_op, ops
from torch import SymInt
from torch._ops import OpOverload
from torch.library import Library


executorch_prims_lib = Library("executorch_prim", "DEF")


@bind_pattern_to_op(executorch_prims_lib, "add.int(SymInt a, SymInt b) -> SymInt")
def add(a: SymInt, b: SymInt) -> SymInt:
    return a + b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "mul.int(SymInt a, SymInt b) -> SymInt")
def mul(a: SymInt, b: SymInt) -> SymInt:
    return a * b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "sub.int(SymInt a, SymInt b) -> SymInt")
def sub(a: SymInt, b: SymInt) -> SymInt:
    return a - b  # pyre-ignore


@bind_pattern_to_op(executorch_prims_lib, "floordiv.int(SymInt a, SymInt b) -> SymInt")
def floordiv(a: SymInt, b: SymInt) -> SymInt:
    return a // b  # pyre-ignore


# TODO: ideally we should return SymBool in the schema, but it seems
# the schema parser does not recognize SymBool yet: P629748075
@bind_pattern_to_op(executorch_prims_lib, "gt.int(SymInt a, SymInt b) -> bool")
def gt(a: SymInt, b: SymInt) -> bool:
    return a > b


@bind_pattern_to_op(executorch_prims_lib, "lt.int(SymInt a, SymInt b) -> bool")
def lt(a: SymInt, b: SymInt) -> bool:
    return a < b


@bind_pattern_to_op(executorch_prims_lib, "ge.int(SymInt a, SymInt b) -> bool")
def ge(a: SymInt, b: SymInt) -> bool:
    return a >= b


@bind_pattern_to_op(executorch_prims_lib, "le.int(SymInt a, SymInt b) -> bool")
def le(a: SymInt, b: SymInt) -> bool:
    return a <= b


@bind_pattern_to_op(executorch_prims_lib, "eq.int(SymInt a, SymInt b) -> bool")
def eq(a: SymInt, b: SymInt) -> bool:
    return a == b


_PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS: Dict[OpOverload, OpOverload] = {
    operator.sub: ops.backend.executorch_prim.sub.int,
    operator.mul: ops.backend.executorch_prim.mul.int,
    operator.add: ops.backend.executorch_prim.add.int,
    operator.floordiv: ops.backend.executorch_prim.floordiv.int,
    operator.eq: ops.backend.executorch_prim.eq.int,
    operator.gt: ops.backend.executorch_prim.gt.int,
    operator.lt: ops.backend.executorch_prim.lt.int,
    operator.ge: ops.backend.executorch_prim.ge.int,
    operator.le: ops.backend.executorch_prim.le.int,
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
