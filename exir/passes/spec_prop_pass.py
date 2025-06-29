# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
from torch.fx.node import Node
from torch.utils import _pytree as pytree
from torch.fx.passes.infra.pass_base import PassResult


# pyre-ignore
def make_spec(x):
    if isinstance(x, torch.Tensor):
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
    # graph signature is None for memory planning passes not called from EdgeProgramManager, these paths are deprecated so mutable buffers are not supported on them.
    if graph_signature is None:
        return False
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                # if the buffer is mutated then record that
                if fqn in graph_signature.buffers_to_mutate.values():
                    return True
    return False
class SpecPropPass:
    def __call__(self, gm: torch.fx.GraphModule) -> PassResult:
        return spec_prop_pass(gm)
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
                    and not _is_mutable_buffer(node, exported_program.graph_signature)
                )
                or node.target
                in exported_program.graph_signature.inputs_to_lifted_tensor_constants
            ):
                spec.const = True

def spec_prop_pass(gm: torch.fx.GraphModule) -> PassResult:
    # Update all the meta["val"]
    pass_result = ExportPass()(gm)
    assert pass_result is not None
    gm = pass_result.graph_module
    # set node.meta["spec"] based on meta["val"]
    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if node.op == "get_attr":
                    continue
                node.meta["spec"] = pytree.tree_map(lambda meta_val: make_spec(meta_val), node.meta["val"])
    return pass_result
