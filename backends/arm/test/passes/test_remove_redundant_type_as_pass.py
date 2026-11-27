# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.remove_redundant_type_as_pass import (
    RemoveRedundantTypeAsPass,
)


class ParameterTypeAs(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.type_as(x)


class InputTypeAs(torch.nn.Module):
    def forward(self, x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return x.type_as(other)


def _export(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    return torch.export.export(module.eval(), inputs, strict=True).module()


def _type_as_count(graph_module: torch.fx.GraphModule) -> int:
    return sum(
        node.target == torch.ops.aten.type_as.default
        for node in graph_module.graph.nodes
    )


def test_removes_same_dtype_parameter_type_as() -> None:
    inputs = (torch.ones(1, 4),)
    graph_module = _export(ParameterTypeAs(torch.float32), inputs)
    assert _type_as_count(graph_module) == 1

    result = RemoveRedundantTypeAsPass()(graph_module)

    assert result is not None
    assert result.modified
    assert _type_as_count(graph_module) == 0


def test_preserves_parameter_type_as_with_dtype_change() -> None:
    inputs = (torch.ones(1, 4, dtype=torch.float16),)
    graph_module = _export(ParameterTypeAs(torch.float32), inputs)

    result = RemoveRedundantTypeAsPass()(graph_module)

    assert result is not None
    assert not result.modified
    assert _type_as_count(graph_module) == 1


def test_preserves_parameter_type_as_with_device_change() -> None:
    inputs = (torch.ones(1, 4, device="meta"),)
    graph_module = _export(ParameterTypeAs(torch.float32), inputs)

    result = RemoveRedundantTypeAsPass()(graph_module)

    assert result is not None
    assert not result.modified
    assert _type_as_count(graph_module) == 1


def test_preserves_runtime_input_type_as() -> None:
    inputs = (torch.ones(1, 4), torch.ones(1, 4))
    graph_module = _export(InputTypeAs(), inputs)

    result = RemoveRedundantTypeAsPass()(graph_module)

    assert result is not None
    assert not result.modified
    assert _type_as_count(graph_module) == 1
