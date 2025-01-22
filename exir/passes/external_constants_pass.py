# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import PassResult
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportedProgram, OutputKind
from torch.fx import GraphModule


def external_constants_pass(
    gm: GraphModule,
) -> PassResult:
    """
    Move all constants to external file.
    """
    mutated = False
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            if node.op == "placeholder":
                spec = node.meta.get("spec")
                if isinstance(spec, TensorSpec) and spec.const:
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
