from executorch.exir.dialects._ops import bind_pattern_to_op
from torch import SymInt
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
