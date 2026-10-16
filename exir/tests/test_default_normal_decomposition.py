# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.tracer import _default_decomposition_table
from torch.export import export


class Normal(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(2.0, 3.0, size=x.shape)


class NormalWithGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.Generator().manual_seed(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(
            2.0,
            3.0,
            size=x.shape,
            generator=self.generator,
        )


class NoGradNormal(torch.nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(2.0, 3.0, size=x.shape)


def test_decompose_normal_float_float() -> None:
    exported_program = export(Normal(), (torch.zeros(2, 3),))

    transformed = exported_program.run_decompositions(_default_decomposition_table())
    targets = {
        node.target for node in transformed.graph.nodes if node.op == "call_function"
    }

    assert torch.ops.aten.normal.float_float not in targets
    assert torch.ops.aten.randn.default in targets
    assert torch.ops.aten.mul.Tensor in targets
    assert torch.ops.aten.add.Tensor in targets


def test_preserve_normal_float_float_with_generator() -> None:
    exported_program = export(NormalWithGenerator(), (torch.zeros(2, 3),))

    transformed = exported_program.run_decompositions(_default_decomposition_table())
    targets = {
        node.target for node in transformed.graph.nodes if node.op == "call_function"
    }

    assert torch.ops.aten.normal.float_float in targets
    assert torch.ops.aten.randn.default not in targets


def test_decompose_normal_float_float_in_no_grad_submodule() -> None:
    exported_program = export(NoGradNormal(), (torch.zeros(2, 3),))

    transformed = exported_program.run_decompositions(_default_decomposition_table())
    targets = {
        node.target for node in transformed.graph.nodes if node.op == "call_function"
    }

    assert torch.ops.aten.normal.float_float not in targets
    assert torch.ops.aten.randn.default in targets
