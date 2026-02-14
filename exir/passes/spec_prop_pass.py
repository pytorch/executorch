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
from executorch.exir.passes.dim_order_utils import (
    dim_order_from_fake_tensor,
    should_propagate_dim_order,
)
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree


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
                    # Ensure every node with val has a spec (base ExportPass may not set it).
                    if "spec" not in node.meta and meta_val is not None:
                        node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                    if node.op == "output":
                        node.meta["spec"] = pytree.tree_map(get_spec, node.args[0])
                    elif node.op == "call_function" and node.target == operator.getitem:
                        value_spec = pytree.tree_map(get_spec, node.args[0])
                        node.meta["spec"] = value_spec[node.args[1]]
                    elif (
                        node.op == "call_function"
                        and should_propagate_dim_order(node.target)
                        and node.args
                    ):
                        # Propagate primary input dim_order for format-preserving ops (Fix #16032).
                        # Handles both clone.out (out= kwarg) and clone.default (single output).
                        self_val = node.args[0].meta.get("val")
                        if self_val is not None:
                            src_dim_order = dim_order_from_fake_tensor(self_val)
                            if "out" in node.kwargs:
                                out_arg = node.kwargs["out"]
                                assert isinstance(
                                    out_arg, torch.fx.Node
                                ), (
                                    f"Expected clone.out 'out' to be fx.Node, got {type(out_arg)}"
                                )
                                out_spec = out_arg.meta.get("spec")
                            else:
                                # clone.default: ensure node has spec (ExportPass may not set it)
                                out_spec = node.meta.get("spec")
                                if out_spec is None and meta_val is not None:
                                    node.meta["spec"] = pytree.tree_map(
                                        make_spec, meta_val
                                    )
                                    out_spec = node.meta["spec"]
                            if (
                                out_spec is not None
                                and hasattr(out_spec, "dim_order")
                                and src_dim_order is not None
                            ):
                                out_spec.dim_order = tuple(src_dim_order)
                    elif (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        if "spec" not in node.meta:
                            node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                        else:
                            node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                        return res

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
