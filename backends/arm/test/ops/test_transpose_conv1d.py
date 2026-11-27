# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import conftest
import pytest
import torch
from executorch.backends.arm.operator_support.convolution_support import (
    ConvolutionSupported,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.mark.parametrize(
    "kwargs,expected_conv_count",
    [
        ({"kernel_size": 4, "stride": 2, "padding": 1}, 1),
        (
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
            },
            1,
        ),
        ({"kernel_size": 5, "padding": 2, "bias": False}, 1),
        ({"kernel_size": 4, "stride": 2, "padding": 1, "groups": 2}, 2),
    ],
    ids=["issue_24697", "output_padding", "no_bias", "grouped"],
)
def test_conv_transpose1d_tosa_FP(kwargs, expected_conv_count):
    """Check full delegation, the TOSA op, and reference-model numerics."""
    model = torch.nn.ConvTranspose1d(16, 8, **kwargs).eval()
    inputs = (torch.randn(1, 16, 32),)
    pipeline = TosaPipelineFP[tuple[torch.Tensor]](
        model,
        inputs,
        "torch.ops.aten.conv_transpose1d.default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.count_tosa_ops({"TRANSPOSE_CONV2D": expected_conv_count})
    pipeline.run()


@common.SkipIfNoModelConverter
def test_conv_transpose1d_vgf():
    """Repro 4 of https://github.com/pytorch/executorch/issues/24697."""
    model = torch.nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1).eval()
    x = torch.randn(1, 16, 32)
    assert model(x).shape == (1, 8, 64)
    ep = torch.export.export(model, (x,))
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[VgfPartitioner(VgfCompileSpec())],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    nodes = [
        node
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function"
    ]
    delegates = [
        node
        for node in nodes
        if node.target == torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(delegates) == 1
    assert all(
        node.target
        in (torch.ops.higher_order.executorch_call_delegate, operator.getitem)
        for node in nodes
    ), [str(node.target) for node in nodes]
    assert edge.to_executorch().buffer


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": 4, "dilation": 2},
        {"kernel_size": 4, "padding": 4},
    ],
    ids=["dilation", "excessive_crop"],
)
def test_conv_transpose1d_rejects_unsupported_options(kwargs):
    """Keep unsupported TOSA dilation and cropping out of the delegate."""
    model = torch.nn.ConvTranspose1d(16, 8, **kwargs).eval()
    ep = to_edge(
        torch.export.export(model, (torch.randn(1, 16, 32),)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    (conv,) = [
        node
        for node in ep.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    checker = ConvolutionSupported(
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        WhyNoPartitionReporter(),
    )
    assert not checker.is_node_supported({}, conv)
