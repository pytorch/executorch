# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.replace_inf_and_limit_values_pass import (
    ReplaceInfAndLimitValuesPass,
)

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from torch import fx


class ModuleWithInf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mask", torch.tensor([float("inf"), float("-inf")], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mask  # type: ignore[operator]
        x = torch.ops.aten.add.Tensor(x, float("-inf"))
        x = torch.ops.aten.add.Tensor(x, float("inf"))
        return x


def _get_add_constants(module_with_infinf: fx.GraphModule) -> list[float]:
    """
    Return the scalar literals passed to `aten.add.Tensor`, skipping tensor inputs.
    """
    return [
        node.args[1]
        for node in module_with_infinf.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.add.Tensor
        and len(node.args) > 1
        and not isinstance(node.args[1], fx.Node)
    ]


def _get_mask_buffer(graph_module: fx.GraphModule) -> torch.Tensor:
    """
    Fetch the `mask` buffer tensor from the traced module.
    """
    buffers = dict(graph_module.named_buffers())
    assert "mask" in buffers, "Mask buffer not found"
    return buffers["mask"]


def test_replace_inf_and_limit_values_no_target_clamps_inf_constants():
    """
    Trace a module with infinities, run ReplaceInfAndLimitValuesPass, and expect the buffer and scalar
    literals to be clamped to ±255 with no infinities left.
    """
    gm = fx.symbolic_trace(ModuleWithInf())

    result = ReplaceInfAndLimitValuesPass().call(gm)
    mask_after_pass = _get_mask_buffer(result.graph_module)

    assert result.modified
    expected = torch.tensor([255.0, -255.0], dtype=mask_after_pass.dtype)
    assert torch.equal(mask_after_pass, expected)
    assert not torch.isinf(mask_after_pass).any()
    assert sorted(_get_add_constants(result.graph_module)) == [-255, 255]


def test_replace_inf_and_limit_values_no_target_respects_disallowed_nodes():
    """
    When nodes opt out of transforms, running the pass in TFA mode should leave the mask buffer
    untouched while still clamping scalar literals to ±255.
    """
    gm = fx.symbolic_trace(ModuleWithInf())
    mask_before = _get_mask_buffer(gm).clone()

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        if any(
            isinstance(arg, fx.Node) and arg.op == "get_attr" and arg.target == "mask"
            for arg in node.args
        ):
            node.meta[DISALLOW_TFA_META_KEY] = True

    replace_inf = ReplaceInfAndLimitValuesPass()
    replace_inf.is_tfa_pass = True

    result = replace_inf.call(gm)
    assert result.modified

    mask_after = _get_mask_buffer(result.graph_module)
    assert torch.equal(mask_after, mask_before)
    assert torch.isinf(mask_after).tolist() == [True, True]
    assert sorted(_get_add_constants(result.graph_module)) == [-255, 255]
