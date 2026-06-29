# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Importing the module registers the channels_last operator dialect.
import executorch.backends.transforms.channels_last_ops  # noqa: F401
import pytest
import torch
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


def _to_nhwc(nchw: torch.Tensor) -> torch.Tensor:
    return nchw.permute(0, 2, 3, 1).contiguous()


def _find(graph_module: torch.fx.GraphModule, target):
    nodes = [
        n
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target == target
    ]
    assert len(nodes) == 1, f"expected exactly one {target}, found {len(nodes)}"
    return nodes[0]


_CONV_CASES = [
    # (N, C_in, H, W, C_out, kernel, stride, padding, dilation, groups, bias)
    (2, 3, 8, 8, 4, 3, 1, 0, 1, 1, True),
    (2, 3, 8, 8, 4, 3, 1, 0, 1, 1, False),
    (1, 4, 10, 10, 6, 3, 2, 1, 1, 1, True),
    (1, 4, 7, 7, 4, 3, 1, 1, 1, 4, True),  # depthwise (groups == C_in == C_out)
]


@pytest.mark.parametrize("n,cin,h,w,cout,k,stride,pad,dil,groups,bias", _CONV_CASES)
def test_convolution_matches_aten(
    n, cin, h, w, cout, k, stride, pad, dil, groups, bias
):
    torch.manual_seed(0)
    nchw = torch.randn(n, cin, h, w)
    weight = torch.randn(cout, cin // groups, k, k)
    bias_t = torch.randn(cout) if bias else None
    nhwc = _to_nhwc(nchw)

    expected = _to_nhwc(
        torch.ops.aten.convolution(
            nchw,
            weight,
            bias_t,
            [stride, stride],
            [pad, pad],
            [dil, dil],
            False,
            [0, 0],
            groups,
        )
    )
    actual = torch.ops.channels_last.convolution(
        nhwc,
        weight,
        bias_t,
        [stride, stride],
        [pad, pad],
        [dil, dil],
        False,
        [0, 0],
        groups,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    "kernel,stride,pad,ceil_mode,count_include_pad",
    [
        (2, 2, 0, False, True),
        (3, 1, 1, False, True),
        (3, 2, 1, True, False),
    ],
)
def test_avg_pool2d_matches_aten(kernel, stride, pad, ceil_mode, count_include_pad):
    torch.manual_seed(0)
    nchw = torch.randn(2, 3, 9, 9)
    nhwc = _to_nhwc(nchw)

    expected = _to_nhwc(
        torch.ops.aten.avg_pool2d(
            nchw,
            [kernel, kernel],
            [stride, stride],
            [pad, pad],
            ceil_mode,
            count_include_pad,
            None,
        )
    )
    actual = torch.ops.channels_last.avg_pool2d(
        nhwc,
        [kernel, kernel],
        [stride, stride],
        [pad, pad],
        ceil_mode,
        count_include_pad,
        None,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("dims", [(0, 3, 1, 2), (0, 2, 3, 1), (3, 2, 1, 0)])
def test_permute_copy_moves_data(dims):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 5, 3)

    expected = torch.ops.aten.permute_copy(x, list(dims))
    actual = torch.ops.channels_last.permute_copy(x, list(dims))

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


def test_convolution_lowers_to_edge_dialect():
    class M(torch.nn.Module):
        def forward(self, x, w, b):
            return torch.ops.channels_last.convolution(
                x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
            )

    nhwc = torch.randn(2, 8, 8, 3)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)

    ep = torch.export.export(M().eval(), (nhwc, weight, bias), strict=True)
    edge = to_edge(ep)

    node = _find(
        edge.exported_program().graph_module,
        exir_ops.edge.channels_last.convolution.default,
    )
    # Fake kernel must yield the correct NHWC output shape (N, H_out, W_out, C_out).
    assert tuple(node.meta["val"].shape) == (2, 6, 6, 4)
