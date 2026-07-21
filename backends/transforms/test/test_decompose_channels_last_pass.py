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


class _AdaptiveAvgPoolModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.adaptive_avg_pool2d(x, [4, 4])


class _UpsampleBilinearModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.upsample_bilinear2d(x, [16, 16], False, None)


class _UpsampleBilinearScaleModule(torch.nn.Module):
    # output_size=None with scale_factors (the other upsample.vec branch).
    def forward(self, x):
        return torch.ops.channels_last.upsample_bilinear2d(x, None, False, [2.0, 2.0])


class _UpsampleNearestModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.upsample_nearest2d(x, [16, 16], None)


class _GridSamplerModule(torch.nn.Module):
    def forward(self, x, grid):
        return torch.ops.channels_last.grid_sampler_2d(x, grid, 0, 0, False)


class _MaxPoolModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.channels_last.max_pool2d_with_indices(
            x, [2, 2], [2, 2], [0, 0], [1, 1], False
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
        _AdaptiveAvgPoolModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.adaptive_avg_pool2d.default,
        exir_ops.edge.aten._adaptive_avg_pool2d.default,
        2,
    ),
    (
        _UpsampleBilinearModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.upsample_bilinear2d.default,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        2,
    ),
    (
        _UpsampleBilinearScaleModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.upsample_bilinear2d.default,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        2,
    ),
    (
        _UpsampleNearestModule(),
        (torch.randn(2, 8, 8, 3),),
        exir_ops.edge.channels_last.upsample_nearest2d.default,
        exir_ops.edge.aten.upsample_nearest2d.vec,
        2,
    ),
    (
        _GridSamplerModule(),
        (torch.randn(2, 8, 8, 3), torch.rand(2, 6, 6, 2) * 2 - 1),
        exir_ops.edge.channels_last.grid_sampler_2d.default,
        exir_ops.edge.aten.grid_sampler_2d.default,
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


# max_pool2d_with_indices is multi-output (values, indices), so it gets dedicated
# tests rather than the single-output _CASES harness.
def test_max_pool2d_with_indices_decomposes():
    args = (torch.randn(2, 8, 8, 3),)
    ep = torch.export.export(_MaxPoolModule().eval(), args, strict=True)
    gm = (
        to_edge(ep)
        .transform([DecomposeChannelsLastPass()])
        .exported_program()
        .graph_module
    )

    assert _count(gm, exir_ops.edge.channels_last.max_pool2d_with_indices.default) == 0
    assert _count(gm, exir_ops.edge.aten.max_pool2d_with_indices.default) == 1
    # permutes: input + values + indices.
    assert _count(gm, exir_ops.edge.aten.permute_copy.default) == 3


def test_max_pool2d_with_indices_decomposed_runs_and_matches_eager():
    module = _MaxPoolModule()
    args = (torch.randn(2, 8, 8, 3),)
    expected_values, expected_indices = module(*args)

    ep = torch.export.export(module.eval(), args, strict=True)
    et = to_edge(ep).transform([DecomposeChannelsLastPass()]).to_executorch()
    method = _load_for_executorch_from_buffer(et.buffer)
    values, indices = method.forward(list(args))

    torch.testing.assert_close(values, expected_values, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(indices, expected_indices)
