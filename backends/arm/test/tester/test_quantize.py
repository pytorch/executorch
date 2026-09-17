# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.tester.quantize import ArmQuantize
from executorch.backends.arm.tosa import TosaSpecification


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def _quantize(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: tuple[dict[int, torch.export.Dim], ...],
) -> torch.fx.GraphModule:
    quantization_config = get_symmetric_quantization_config()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    stage = ArmQuantize(
        quantizer,
        quantization_config,
        dynamic_shapes=dynamic_shapes,
    )

    stage.run(module, inputs)  # type: ignore[arg-type]

    return stage.artifact


def test_arm_quantize_preserves_dynamic_input_shape() -> None:
    inputs = (torch.randn(2, 4),)
    batch = torch.export.Dim("batch", min=1, max=4)

    graph_module = _quantize(torch.nn.ReLU(), inputs, ({0: batch},))

    placeholder = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    assert isinstance(placeholder.meta["val"].shape[0], torch.SymInt)
    assert graph_module(torch.randn(3, 4)).shape == (3, 4)


def test_arm_quantize_preserves_shared_dynamic_input_shape() -> None:
    inputs = (torch.randn(2, 4), torch.randn(2, 4))
    batch = torch.export.Dim("batch", min=1, max=4)

    graph_module = _quantize(Add(), inputs, ({0: batch}, {0: batch}))

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    batch_sizes = [node.meta["val"].shape[0] for node in placeholders]
    assert all(isinstance(size, torch.SymInt) for size in batch_sizes)
    assert batch_sizes[0] == batch_sizes[1]
    assert graph_module(torch.randn(3, 4), torch.randn(3, 4)).shape == (3, 4)
