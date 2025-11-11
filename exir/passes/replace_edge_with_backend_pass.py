# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.passes.executorch_prim_ops_registry import (
    _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS,
)
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class EdgeToBackendOpsPass(PassBase):
    """
    Converts
    1. symbolic int ops to the executorch_prims namespaced ops
    2. other backend ops from torch._ops.OpOverload to BackendOpOverload
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            for node in module.graph.nodes:
                if node.op == "call_function":
                    if node.target in _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS:
                        node.target = _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS[node.target]

                    elif isinstance(node.target, torch._ops.OpOverload):
                        # replace torch.ops.OpOverload with its corresponding backend ops.
                        # Looking op name up from _dir in _DialectNamespace, _OpNamespace
                        # and BackendOpOverloadPacket

                        def get_new_op(
                            target: torch._ops.OpOverload,
                        ) -> torch._ops.OpOverload:
                            namespace = target.namespace
                            name = target._schema.name.split("::")[1]
                            overload_name = target._overloadname
                            obj = ops.backend
                            for key in [namespace, name, overload_name]:
                                if key not in obj._dir:
                                    return target
                                obj = getattr(obj, key)
                            return obj

                        node.target = get_new_op(node.target)

            module.recompile()

        return PassResult(graph_module, True)
