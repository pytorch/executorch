# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Optional

import torch
from executorch.exir.pass_base import PassResult
from executorch.exir.tensor import TensorSpec

from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export.exported_program import ExportedProgram, OutputKind
from torch.fx import GraphModule


def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )


def external_constants_pass(
    gm: GraphModule,
    gen_tag_fn: Optional[Callable[[torch.fx.Node], Optional[str]]] = None,
) -> PassResult:
    """
    Move all non-lifted constants to external file.
    NOTE: Lifted constants are not moved as they are closer
    to code than data.
    """
    mutated = False
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            if (node.op == "placeholder") and ("_lifted_tensor" not in node.name):
                spec = node.meta.get("spec")
                if isinstance(spec, TensorSpec) and spec.const:
                    if gen_tag_fn is not None:
                        node.meta["constant_tag"] = gen_tag_fn(node)
                    else:
                        node.meta["constant_tag"] = "_default_external_constant"
                    mutated = True
    return PassResult(gm, mutated)


def _is_mutable_weight(node: torch.fx.Node, ep: ExportedProgram) -> bool:
    grad_targets = [
        spec.target
        for spec in ep.graph_signature.output_specs
        if spec.kind == OutputKind.GRADIENT_TO_PARAMETER
    ]
    return (
        node.op == "placeholder"
        and node.target in ep.graph_signature.inputs_to_parameters.keys()
        and ep.graph_signature.inputs_to_parameters[node.target] in grad_targets
    )


def external_mutable_weights_pass(
    gm: GraphModule,
    ep: ExportedProgram,
) -> PassResult:
    """
    Move all mutable weights to external file.
    """
    # pass the gm and the ep seperately as the gm is being mutated by a bunch of passes in to_executorch,
    # so the gm in the ep is lagging the graph signature is still correct.
    # This is really tech debt and all the passes should be refactored to just mutate the ep.
    mutated = False
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            if node.op == "placeholder":
                spec = node.meta.get("spec")
                if (
                    isinstance(spec, TensorSpec)
                    and spec.const
                    and _is_mutable_weight(node, ep)
                ):
                    node.meta["constant_tag"] = "_default_external_constant"
                    mutated = True
    return PassResult(gm, mutated)


# Note: this pass must be run on an unlifted graph, e.g. ep.module(),
# and not on a lifted graph, e.g. ep.graph_module.
# This is using 'get_attr' to tag constants, which only appears in
# unlifted graphs.
def delegate_external_constants_pass_unlifted(
    module: torch.nn.Module,
    gen_tag_fn: Optional[Callable[[torch.fx.Node], Optional[str]]] = None,
) -> PassResult:
    mutated = False
    for m in module.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for node in m.graph.nodes:
            if node.op == "get_attr":
                if gen_tag_fn is not None:
                    node.meta.setdefault("custom", {})
                    node.meta["custom"]["delegate_constant_tag"] = gen_tag_fn(node)
                    mutated = True
    return PassResult(module, mutated)
