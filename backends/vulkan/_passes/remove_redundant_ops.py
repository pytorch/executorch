# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set, Union

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.tensor import TensorSpec

OpType = Union[str, torch._ops.OpOverload, EdgeOpOverload]


class RemoveRedundantOpsTransform(ExportPass):
    """
    Trim certain operators to reduce unnecessary overhead.
    """

    redundant_ops: Set[OpType] = {
        torch.clone,
        torch.ops.aten.clone.default,
        exir_ops.edge.aten.clone.default,
        torch.ops.aten.alias.default,
        exir_ops.edge.aten.alias.default,
        torch.ops.aten.alias_copy.default,
        exir_ops.edge.aten.alias_copy.default,
        exir_ops.edge.aten.lift_fresh_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.aten.expand_copy.default,
        # copy.default(self, src): no-op when src dtype/shape matches self.
        exir_ops.edge.aten.copy.default,
        # view_copy.default: a view onto the same shape. These survive into the
        # partition and dispatch a shader that reads every texel and writes it
        # back unchanged. Only same-shape views are safe to drop here: a view
        # that changes the rank must keep its node, because the ops downstream
        # read their argument's shape.
        exir_ops.edge.aten.view_copy.default,
    }

    # For these ops the meaningful input is args[1] (src), not args[0] (self).
    _src_arg1_ops: Set[OpType] = {
        exir_ops.edge.aten.copy.default,
    }

    def __init__(self) -> None:
        super(RemoveRedundantOpsTransform, self).__init__()

    def _should_remove(self, node: torch.fx.Node) -> bool:
        if node.target not in self.redundant_ops:
            return False

        src_arg_idx = 1 if node.target in self._src_arg1_ops else 0
        orig_node = node.args[src_arg_idx]
        assert isinstance(orig_node, torch.fx.Node)

        src_dtype = orig_node.meta["val"].dtype
        dst_dtype = node.meta["val"].dtype

        # Do not remove if the op is converting the dtype.
        if src_dtype != dst_dtype:
            return False

        src_shape = orig_node.meta["val"].shape
        dst_shape = node.meta["val"].shape

        return src_shape == dst_shape

    def _remove(self, graph_module: torch.fx.GraphModule) -> None:
        for node in graph_module.graph.nodes:
            if not self._should_remove(node):
                continue

            src_arg_idx = 1 if node.target in self._src_arg1_ops else 0
            node.replace_all_uses_with(node.args[src_arg_idx])

        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)


class FoldForceFp16CastsPass(ExportPass):
    """
    Under `force_fp16` every fp32 tensor runs as fp16, so a cast between the two
    copies identical data (a Blit_Half_to_Half per call). Consumers of an
    fp16 -> fp32 cast read its fp16 input directly, except a graph output, which
    keeps the cast. An fp32 -> fp16 cast is dropped by marking its producer
    fp16. That changes nothing for an intermediate tensor, which runs as fp16
    either way, but it would change the staging dtype of a graph input or
    output, so producers that are user inputs or graph outputs keep the cast.
    Runs after SpecPropPass, which would reject the mixed dtypes this leaves
    behind.
    """

    _casts: Set[OpType] = {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }
    _pair: Set[torch.dtype] = {torch.float16, torch.float32}

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in self._casts:
                continue
            src = node.args[0]
            if not isinstance(src, torch.fx.Node):
                continue
            src_spec = src.meta.get("spec")
            if not isinstance(src_spec, TensorSpec):
                continue
            dst_dtype = node.meta["val"].dtype
            # An earlier fold may have retyped src's spec to fp16; meta["val"]
            # keeps the traced dtype.
            traced = {src.meta["val"].dtype, dst_dtype}
            if self._pair not in (traced, {src_spec.dtype, dst_dtype}):
                continue
            if dst_dtype == torch.float16:
                if src_spec.dtype != torch.float16:
                    if (src.op == "placeholder" and not src_spec.const) or any(
                        user.op == "output" for user in src.users
                    ):
                        continue
                    src_spec.scalar_type = torch.float16
                node.replace_all_uses_with(src)
            else:
                if all(user.op == "output" for user in node.users):
                    continue
                node.replace_all_uses_with(
                    src, delete_user_cb=lambda user: user.op != "output"
                )
            if not node.users:
                graph_module.graph.erase_node(node)
            modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
