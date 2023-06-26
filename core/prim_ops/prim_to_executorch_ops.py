import operator
from typing import Dict, Set

# necessary to ensure the ops are registered
import executorch.core.prim_ops.executorch_prim_ops_registry  # noqa
import torch
from executorch.exir.dialects._ops import ops
from torch._ops import OpOverload


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
