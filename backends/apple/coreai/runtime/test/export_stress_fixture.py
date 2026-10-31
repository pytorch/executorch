# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import operator
from pathlib import Path

import torch
from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.partition.partitioner import do_not_delegate
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_backend_modules,
)
from executorch.exir.pass_base import PassResult


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(64, 128)
        self.down = torch.nn.Linear(128, 64)

    def forward(self, x):
        return torch.relu(x + self.down(torch.relu(self.up(x))))


class ResidualMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(32, 64)
        self.blocks = torch.nn.Sequential(*(ResidualBlock() for _ in range(3)))
        self.output = torch.nn.Linear(64, 16)

    def forward(self, x):
        return self.output(self.blocks(torch.relu(self.input(x))))


class TagResidualBoundary:
    def __call__(self, graph_module):
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.relu.default
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target == exir_ops.edge.aten.add.Tensor
            ):
                do_not_delegate(node)
                return PassResult(graph_module, True)
        raise RuntimeError("Missing residual boundary")


def export_fixture(output_dir):
    # Distinct weights keep SDK artifacts separate from the playground model.
    torch.manual_seed(20261029)
    model = ResidualMLP().eval().half().requires_grad_(False)
    sample = torch.randn(1, 32, dtype=torch.float16)
    with torch.no_grad():
        expected = model(sample)
    assert expected.dtype == torch.float16 and torch.isfinite(expected).all()
    exported = torch.export.export(model, (sample,))
    lowered = to_edge_transform_and_lower(
        exported,
        transform_passes=[*get_default_passes(), TagResidualBoundary()],
        partitioner=[CoreAIPartitioner()],
        compile_config=get_default_compile_config(),
    )
    graph = lowered.exported_program().graph_module
    delegates = get_lowered_backend_modules(graph)
    assert len(delegates) == 2 and all(
        d.backend_id == "CoreAIBackend" for d in delegates
    )
    allowed = {executorch_call_delegate, operator.getitem}
    fallback = [
        node
        for node in graph.graph.nodes
        if node.op in {"call_function", "call_method", "call_module"}
        and (node.op != "call_function" or node.target not in allowed)
    ]
    assert len(fallback) == 1 and fallback[0].target == exir_ops.edge.aten.relu.default
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pte").open("wb") as stream:
        lowered.to_executorch().write_to_file(stream)
    (output_dir / "fixture.json").write_text(
        json.dumps(
            {
                "dtype": "float16",
                "input_shape": list(sample.shape),
                "output_shape": list(expected.shape),
                "input": sample.float().flatten().tolist(),
                "expected": expected.float().flatten().tolist(),
            }
        )
        + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    export_fixture(parser.parse_args().output_dir)
