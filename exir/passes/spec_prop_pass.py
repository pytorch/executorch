# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Optional

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.pass_base import ExportPass, ProxyValue
from executorch.exir.tensor import TensorSpec, dim_order_from_stride
from torch.export.exported_program import ExportGraphSignature
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree


# Ops that TRANSFORM layout — output dim_order from op's explicit dim_order kwarg.
# Source: ExecuTorch issues #8037 and #6330. Verified (Q3): torch.ops.dim_order_ops.
try:
    _LAYOUT_TRANSFORMING_OPS = frozenset({
        torch.ops.dim_order_ops._to_dim_order_copy.default,
        torch.ops.dim_order_ops._clone_dim_order.default,
    })
except AttributeError:
    _LAYOUT_TRANSFORMING_OPS = frozenset()

# Ops where output memory format is IDENTICAL to primary input. Reference: PyTorch docs memory_format.
_FORMAT_PRESERVING_OPS: frozenset = frozenset({
    torch.ops.aten.clone.default,
    torch.ops.aten.clone.out,
    torch.ops.aten.relu.default,
    torch.ops.aten.relu.out,
    torch.ops.aten.relu_.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu.out,
    torch.ops.aten.silu_.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.gelu.out,
    torch.ops.aten.neg.default,
    torch.ops.aten.neg.out,
    torch.ops.aten.abs.default,
    torch.ops.aten.abs.out,
    torch.ops.aten.exp.default,
    torch.ops.aten.exp.out,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sqrt.out,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.rsqrt.out,
})


def _get_primary_tensor_input(node: Node) -> Optional[Node]:
    """First argument that is an fx.Node with a FakeTensor val (primary input for layout)."""
    for arg in node.args:
        if (
            isinstance(arg, Node)
            and isinstance(arg.meta.get("val"), torch.Tensor)
        ):
            return arg
    return None


def _fix_out_spec_dim_order(node: Node) -> None:
    """
    For out-variant nodes, set the out kwarg node's TensorSpec.dim_order to the
    layout the op will produce. Fixes Code=18 at runtime (issue #16032).
    """
    out_node = node.kwargs.get("out")
    if not isinstance(out_node, Node):
        return
    spec = out_node.meta.get("spec")
    if spec is None:
        return
    if node.target in _LAYOUT_TRANSFORMING_OPS:
        explicit_dim_order = node.kwargs.get("dim_order")
        if explicit_dim_order is not None:
            spec.dim_order = list(int(d) for d in explicit_dim_order)
    elif node.target in _FORMAT_PRESERVING_OPS:
        primary = _get_primary_tensor_input(node)
        if primary is None:
            return
        input_val = primary.meta.get("val")
        if not isinstance(input_val, torch.Tensor):
            return
        spec.dim_order = dim_order_from_stride(input_val)


# pyre-ignore
def make_spec(x):
    if isinstance(x, ProxyValue):
        return make_spec(x.node.meta["val"])
    elif isinstance(x, torch.Tensor):
        return TensorSpec.from_tensor(x)
    elif isinstance(x, (int, bool, float)):
        return x
    else:
        return None


def _is_mutable_buffer(
    node: Node, graph_signature: Optional[ExportGraphSignature] = None
) -> bool:
    """
    Check if the node is mutable buffer according to the provided graph signature.
    """
    if graph_signature is None:
        return False
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                if fqn in graph_signature.buffers_to_mutate.values():
                    return True
    return False


class SpecPropPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Re-trace metadata to ensure it's up to date.
        res = ExportPass()(graph_module)
        assert res is not None
        gm = res.graph_module

        def get_spec(x):
            if hasattr(x, "meta"):
                return x.meta.get("spec", None)
            else:
                return None

        for module in gm.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    meta_val = node.meta.get("val", None)
                    if node.op == "output":
                        node.meta["spec"] = pytree.tree_map(get_spec, node.args[0])
                    elif node.op == "call_function" and node.target == operator.getitem:
                        value_spec = pytree.tree_map(get_spec, node.args[0])
                        node.meta["spec"] = value_spec[node.args[1]]
                    elif (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        if "spec" not in node.meta:
                            node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                        else:
                            node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                        continue
                    if "spec" not in node.meta and meta_val is not None:
                        node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                    _fix_out_spec_dim_order(node)

        return res

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return self(graph_module)

    def update_placeholder_tensor_specs(
        self,
        exported_program: torch.export.ExportedProgram,
        graph_module: torch.fx.GraphModule,
    ) -> None:
        """
        Update the tensor specs for all placeholder nodes such that
        placeholders that are parameters are marked as constant.
        """
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "spec" not in node.meta:
                raise RuntimeError(f"Placeholder node {node} missing meta['spec']")
            spec = node.meta["spec"]
            if isinstance(node.target, str) and (
                node.target in exported_program.graph_signature.inputs_to_parameters
                or (
                    node.target in exported_program.graph_signature.inputs_to_buffers
                    and not _is_mutable_buffer(
                        node, exported_program.graph_signature
                    )
                )
                or node.target
                in exported_program.graph_signature.inputs_to_lifted_tensor_constants
            ):
                spec.const = True
