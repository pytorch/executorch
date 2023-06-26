# pyre-strict

import torch
from executorch.core.prim_ops.prim_to_executorch_ops import (
    _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS,
)
from executorch.exir.dialects._ops import ops
from executorch.exir.pass_base import ExportPass


class EdgeToBackendOpsPass(ExportPass):
    """
    Converts
    1. symbolic int ops to the executorch_prims namespaced ops
    2. other backend ops from torch._ops.OpOverload to BackendOpOverload
    """

    # pyre-ignore
    def call_sym(self, target, args, meta):
        if target in _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS:
            return super().call_operator(
                _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS[target],
                args,
                {},
                meta,
            )
        return super().call_sym(target, args, meta)

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        # replace torch.ops.OpOverload with its corresponding backend ops.
        # Looking op name up from _dir in _DialectNamespace, _OpNamespace
        # and BackendOpOverloadPacket
        if isinstance(op, torch._ops.OpOverload):
            namespace = op.namespace
            name = op._schema.name.split("::")[1]
            overload_name = op._overloadname
            obj = ops.backend
            for key in [namespace, name, overload_name]:
                if key not in obj._dir:
                    return super().call_operator(op, args, kwargs, meta)
                obj = getattr(obj, key)
            return super().call_operator(obj, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)
