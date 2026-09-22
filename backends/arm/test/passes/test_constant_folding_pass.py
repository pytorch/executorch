# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from executorch.backends.arm._passes.constant_folding_pass import ConstantFoldingPass


class ReadOnlyBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.ones(2, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        return x + state[0]


class MutableBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.ones(2, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        state_0 = state[0]
        state_1 = state[1]
        new_state = torch.stack([state_0 + x, state_1 + x])
        state.copy_(new_state)
        return state_0 + x


def _export_module(module: torch.nn.Module) -> torch.fx.GraphModule:
    return torch.export.export(
        module.eval(),
        (torch.ones(1, 4),),
        strict=False,
    ).module()


def _contains_target(graph_module: torch.fx.GraphModule, target) -> bool:
    return any(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def test_constant_folding_folds_read_only_buffer() -> None:
    graph_module = _export_module(ReadOnlyBufferModel())

    result = ConstantFoldingPass()(graph_module)

    assert result is not None
    assert result.modified
    assert not _contains_target(graph_module, torch.ops.aten.select.int)
    assert any(
        name.startswith("_frozen_param") for name, _ in graph_module.named_buffers()
    )


def test_constant_folding_preserves_mutable_buffer_read() -> None:
    graph_module = _export_module(MutableBufferModel())

    result = ConstantFoldingPass()(graph_module)

    assert result is not None
    assert _contains_target(graph_module, torch.ops.aten.copy_.default)
    assert _contains_target(graph_module, torch.ops.aten.select.int)
    assert not any(
        name.startswith("_frozen_param") for name, _ in graph_module.named_buffers()
    )
