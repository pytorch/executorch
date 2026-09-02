# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.decompose_add_sub_alpha_pass import (
    DecomposeAddSubAlphaPass,
)
from executorch.backends.arm._passes.scalars_to_attribute_pass import (
    ScalarsToAttributePass,
)
from executorch.backends.arm.test import common


def rewrite(module, inputs):
    return (
        ScalarsToAttributePass()
        .call(torch.export.export(module, inputs).module())
        .graph_module
    )


def lower_alpha(module, inputs):
    """Run the two passes in pipeline order.

    The decomposition folds alpha away first, so the scalar rewrite only ever
    sees an rsub whose alpha is one.

    """
    exported = torch.export.export(module, inputs).module()
    folded = DecomposeAddSubAlphaPass().call(exported).graph_module
    return ScalarsToAttributePass().call(folded).graph_module


def nodes_with_target(graph_module, target):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is target
    ]


def subs(graph_module):
    return nodes_with_target(graph_module, torch.ops.aten.sub.Tensor)


def carries_alpha(node):
    """Rsub takes alpha positionally, add and sub keep it keyword-only."""
    if node.target is torch.ops.aten.rsub.Scalar:
        return len(node.args) > 2
    return "alpha" in node.kwargs


class Rsub(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1.0, alpha=self.alpha)


alpha_test_data = {
    "default": 1,
    "positive": 2,
    "negative": -2,
    "fractional": 0.5,
}


@common.parametrize("alpha", alpha_test_data)
def test_rsub_alpha_is_folded_into_a_multiply(alpha) -> None:
    """Alpha scales the operand the swap moves, so it becomes a multiply.

    Nothing may reach the backend still carrying it: the TOSA operators take
    two arguments and reject a third.

    """
    module = Rsub(alpha).eval()
    inputs = (torch.tensor([0.7, -0.25, 2.0]),)
    rewritten = lower_alpha(module, inputs)

    assert len(subs(rewritten)) == 1
    assert not any(
        carries_alpha(node)
        for node in rewritten.graph.nodes
        if node.op == "call_function"
    )
    muls = nodes_with_target(rewritten, torch.ops.aten.mul.Tensor)
    assert len(muls) == (0 if alpha == 1 else 1)
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))


class TwoRsubs(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1.0, alpha=2) + torch.rsub(x, 2.0, alpha=3)


def test_each_rsub_is_rewritten_once() -> None:
    """The rewrite reads the whole argument list, so running it per converted
    argument would emit one sub per scalar rather than one per rsub.
    """
    module = TwoRsubs().eval()
    inputs = (torch.tensor([0.7, -0.25, 2.0]),)
    rewritten = lower_alpha(module, inputs)

    assert len(subs(rewritten)) == 2
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))


class IntRsub(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1)


def test_an_all_integer_rsub_is_left_alone() -> None:
    """An integer scalar over an integer tensor converts nothing, so the op
    keeps its own spelling rather than becoming a sub.
    """
    module = IntRsub().eval()
    inputs = (torch.tensor([1, 2, 3], dtype=torch.int32),)
    rewritten = rewrite(module, inputs)

    targets = [
        node.target for node in rewritten.graph.nodes if node.op == "call_function"
    ]
    assert targets == [torch.ops.aten.rsub.Scalar]
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))
