# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from torch.export.experimental import _export_forward_backward
from torch.export.graph_signature import InputKind, InputSpec, TensorArgument


class TensorStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))
        self.register_buffer("persistent", torch.full((4, 4), 2.0))
        self.register_buffer("temporary", torch.full((4, 4), 3.0), persistent=False)
        self.constant = torch.full((4, 4), 4.0)
        self.register_buffer("updated", torch.zeros(4, 4))
        self.register_buffer("kept", torch.tensor(1.0))

    def forward(self, x, unused_input):
        self.updated.copy_(x)
        return (
            x
            + self.weight
            + self.persistent
            + self.temporary
            + self.constant
            + self.kept
        )


def test_removes_unused_state_and_preserves_mutation():
    inputs = (torch.randn(4, 4), torch.randn(1))
    ep = torch.export.export(TensorStateModel(), inputs).run_decompositions()
    nodes = {node.name: node for node in ep.graph.nodes}
    removed_targets = set()
    for spec in ep.graph_signature.input_specs:
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        ) and spec.target not in ("updated", "kept"):
            nodes[spec.arg.name].replace_all_uses_with(nodes["x"])
            removed_targets.add(spec.target)
    ep.graph_module.recompile()
    assert removed_targets == {"weight", "persistent", "temporary", "constant"}
    assert not nodes["b_updated"].users
    reference = copy.deepcopy(ep).module()
    shared_state = ep.state_dict
    shared_constants = ep.constants

    result = RemoveUnusedConstantsPass()(ep)
    assert result.modified
    ep.validate()
    assert set(ep.state_dict) == {"updated", "kept"}
    assert not ep.constants
    assert ep.graph_signature.user_inputs == ("x", "unused_input")
    assert "updated" in ep.graph_signature.buffers_to_mutate.values()
    assert set(shared_state) == {"weight", "persistent", "updated", "kept"}
    assert set(shared_constants) == {"temporary", "constant"}
    assert ep.state_dict["updated"] is shared_state["updated"]
    assert ep.state_dict["kept"] is shared_state["kept"]
    actual = ep.module()
    for _ in range(2):
        inputs = (torch.randn(4, 4), torch.randn(1))
        torch.testing.assert_close(actual(*inputs), reference(*inputs))
        torch.testing.assert_close(actual.updated, inputs[0])
    assert not RemoveUnusedConstantsPass()(ep).modified


def test_preserves_parameter_with_gradient_output():
    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, x):
            return (x * self.weight).sum()

    ep = _export_forward_backward(torch.export.export(Loss(), (torch.ones(2, 2),)))
    nodes = {node.name: node for node in ep.graph.nodes}
    nodes["p_weight"].replace_all_uses_with(nodes["x"])
    ep.graph_module.recompile()
    assert not nodes["p_weight"].users
    assert not RemoveUnusedConstantsPass()(ep).modified
    ep.validate()
    assert ep.graph_signature.parameters == ("weight",)
    assert "weight" in ep.state_dict


def test_preserves_storage_referenced_by_another_placeholder():
    model = torch.nn.Linear(2, 2, bias=False)
    inputs = (torch.randn(1, 2),)
    ep = torch.export.export(model, inputs)
    weight = next(node for node in ep.graph.nodes if node.name == "p_weight")
    with ep.graph.inserting_before(weight):
        unused = ep.graph.placeholder("unused_weight")
    unused.meta = weight.meta.copy()
    ep.graph_signature.input_specs.insert(
        0, InputSpec(InputKind.PARAMETER, TensorArgument(unused.name), "weight")
    )
    ep.graph_module.recompile()
    ep.validate()
    original = ep.state_dict["weight"]
    assert RemoveUnusedConstantsPass()(ep).modified
    ep.validate()
    assert ep.state_dict["weight"] is original
    torch.testing.assert_close(ep.module()(*inputs), model(*inputs))


def test_preserves_constant_in_module_call_signature():
    class Inner(torch.nn.Module):
        def forward(self, x, weight):
            return x + 1

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.inner = Inner()

        def forward(self, x):
            return self.inner(x, self.weight)

    model = Outer()
    inputs = (torch.randn(2, 2),)
    ep = torch.export.export(model, inputs, preserve_module_call_signature=("inner",))
    ep.validate()
    weight = next(node for node in ep.graph.nodes if node.name == "p_weight")
    assert not weight.users
    assert not RemoveUnusedConstantsPass()(ep).modified
    ep.validate()
    torch.testing.assert_close(ep.module()(*inputs), model(*inputs))
