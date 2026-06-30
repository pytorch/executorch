# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Importing registers the channels_last dialect.
import executorch.backends.transforms.channels_last_ops  # noqa: F401
import pytest
import torch
from executorch.backends.transforms.decompose_channels_last_pass import (
    DecomposeChannelsLastPass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


def _count(graph_module: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target == target
    )


class _ConvModule(torch.nn.Module):
    def forward(self, x, w, b):
        return torch.ops.channels_last.convolution(
            x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
        )


class _DepthwiseConvModule(torch.nn.Module):
    # Depthwise (groups == C) and no bias.
    def forward(self, x, w):
        return torch.ops.channels_last.convolution(
            x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3
        )


class _AvgPoolModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.avg_pool2d(
            x, [2, 2], [2, 2], [0, 0], False, True, None
        )


class _PermuteModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.permute_copy(x, [0, 3, 1, 2])


_CASES = [
    (
        _ConvModule(),
        (torch.randn(2, 8, 8, 3), torch.randn(4, 3, 3, 3), torch.randn(4)),
        exir_ops.edge.channels_last.convolution.default,
        exir_ops.edge.aten.convolution.default,
        2,
    ),
    (
        _DepthwiseConvModule(),
        (torch.randn(1, 8, 8, 3), torch.randn(3, 1, 3, 3)),
        exir_ops.edge.channels_last.convolution.default,
        exir_ops.edge.aten.convolution.default,
        2,
    ),
    (
        _AvgPoolModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.avg_pool2d.default,
        exir_ops.edge.aten.avg_pool2d.default,
        2,
    ),
    (
        _PermuteModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.permute_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        1,
    ),
]


@pytest.mark.parametrize("module,args,cl_op,aten_op,n_permutes", _CASES)
def test_decomposes_to_aten_and_permutes(module, args, cl_op, aten_op, n_permutes):
    ep = torch.export.export(module.eval(), args, strict=True)
    edge = to_edge(ep).transform([DecomposeChannelsLastPass()])
    gm = edge.exported_program().graph_module

    assert _count(gm, cl_op) == 0
    assert _count(gm, aten_op) == 1
    assert _count(gm, exir_ops.edge.aten.permute_copy.default) == n_permutes


@pytest.mark.parametrize("module,args,cl_op,aten_op,n_permutes", _CASES)
def test_decomposed_program_runs_and_matches_eager(
    module, args, cl_op, aten_op, n_permutes
):
    eager = module(*args)

    ep = torch.export.export(module.eval(), args, strict=True)
    et = to_edge(ep).transform([DecomposeChannelsLastPass()]).to_executorch()
    method = _load_for_executorch_from_buffer(et.buffer)
    actual = method.forward(list(args))[0]

    torch.testing.assert_close(actual, eager, atol=1e-4, rtol=1e-4)
