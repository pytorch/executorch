# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ArmPassManager, DecomposeMaxPool1dPass
from executorch.backends.arm.test import common


class _MaxPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()
        self.pool = torch.nn.MaxPool1d(kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def test_decompose_max_pool1d_to_max_pool2d() -> None:
    for kernel_size, stride, padding in [(4, 2, 0), (3, 2, 1), (3, 1, 0)]:
        module = _MaxPool1d(kernel_size, stride, padding).eval()
        inputs = (torch.randn(1, 8, 32),)
        exported = torch.export.export(module, inputs, strict=True)

        result = DecomposeMaxPool1dPass()(exported.graph_module)
        assert result is not None
        graph_module = result.graph_module
        graph_module.graph.lint()

        targets = {
            node.target
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        }
        assert torch.ops.aten.max_pool1d.default not in targets
        assert torch.ops.aten.max_pool2d.default in targets

        actual = graph_module(*inputs)
        if isinstance(actual, tuple):
            (actual,) = actual
        torch.testing.assert_close(module(*inputs), actual)


def test_pass_skips_graph_with_submodules_but_no_targets() -> None:
    """Graphs with call_module nodes but no max_pool1d must not retrace.

    ExportPass retracing raises on call_module nodes; the targeted pre-scan must
    skip such graphs (e.g. stateful LSTM prepare graphs) entirely.

    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_module("lin", args=(x,))
    graph.output(out)
    graph_module = torch.fx.GraphModule({"lin": torch.nn.Linear(8, 8)}, graph)

    result = DecomposeMaxPool1dPass()(graph_module)

    assert result is not None
    assert not result.modified


def test_max_pool1d_decomposed_in_tfa_pipeline() -> None:
    module = _MaxPool1d(4, 2).eval()
    inputs = (torch.randn(1, 16, 50),)
    exported = torch.export.export(module, inputs, strict=True)

    transformed = ArmPassManager(
        common.get_tosa_compile_spec("TOSA-1.0+INT")
    ).transform_for_annotation_pipeline(exported.graph_module)

    targets = {
        node.target for node in transformed.graph.nodes if node.op == "call_function"
    }
    assert torch.ops.aten.max_pool1d.default not in targets
    assert torch.ops.aten.max_pool2d.default in targets
