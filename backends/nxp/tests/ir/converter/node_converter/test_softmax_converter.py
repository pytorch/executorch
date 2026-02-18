# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import SoftmaxModule
from executorch.exir.dialects._ops import ops as exir_ops

# noinspection PyProtectedMember
ExecutorchDelegateCall = torch._higher_order_ops.executorch_call_delegate
Softmax = exir_ops.edge.aten._softmax.default


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ConvSoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 1)
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return self.softmax(x)


def assert_softmax_delegated(graph):
    assert graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(graph, [Softmax])


def assert_softmax_not_delegated(graph):
    assert not graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(graph, [Softmax])


def random_input_data(input_shape):
    return (np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0).astype(
        np.int8
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        # Dim must always be the last dimension, which must be a multiple of 8 (num_macs).
        pytest.param((4096, 128), -1, id="2D_total_size_limit"),
        pytest.param((5, 8), -1, id="2D_dim_-1"),
        pytest.param((5, 8), 1, id="2D_dim_1"),
        pytest.param((4096, 8), -1, id="2D_WxH_limit"),
        pytest.param((2, 2048 - 8), -1, id="2D_channels_limit"),
        pytest.param((5, 4, 8), -1, id="3D_dim_-1"),
        pytest.param((4096, 1, 8), -1, id="3D_WxH_limit"),
        pytest.param((5, 4, 3, 8), -1, id="4D_dim_-1"),
        pytest.param((1, 64, 64, 8), -1, id="4D_WxH_limit"),
        pytest.param((64, 1, 64, 128), -1, id="4D_total_size_limit"),
        pytest.param((5, 4, 3, 2, 8), -1, id="5D_dim_-1"),
    ],
)
def test_softmax_delegation(input_shape, dim: int, mocker):
    model = SoftmaxModule(dim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    assert_softmax_delegated(delegated_ep.graph)

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return
    input_data = random_input_data(input_shape)

    # Make sure the tested program contains the `softmax`, and its input has the expected rank.
    nodes = list(intermediate_ep.graph.nodes)
    assert nodes[2].target == Softmax
    assert len(nodes[2].args[0].meta["val"].shape) == len(input_shape)

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
    )


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        # `dim` must be the second dimension, which must be a multiple of 8 (num_macs).
        pytest.param((1, 8, 2, 3), 1, id="4D_dim_1"),
        pytest.param((1, 8, 64, 64), 1, id="4D_WxH_limit"),
        pytest.param((64, 128, 1, 64), -3, id="4D_dim_-3_total_size_limit"),
    ],
)
def test_softmax_delegation__channel_first(input_shape, dim: int, mocker):
    model = ConvSoftmaxModule(dim, input_shape[1])

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    assert_softmax_delegated(delegated_ep.graph)

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return
    input_data = random_input_data(input_shape)

    # Make sure the tested program contains the `softmax`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [Softmax])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        # `dim` is not the last dimension.
        pytest.param((10, 32), 0, id="2D_dim_0"),
        pytest.param((10, 32, 32), 1, id="3D_dim_1"),
        pytest.param((10, 32, 32, 8), 2, id="4D_dim_2"),
        pytest.param((10, 32, 32, 8, 8), 3, id="5D_dim_3"),
        pytest.param((10, 32, 32, 8, 8), 2, id="5D_dim_2"),
    ],
)
def test_softmax_delegation__unsupported_dims(input_shape, dim: int):
    model = SoftmaxModule(dim)
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()
    assert_softmax_not_delegated(delegated_ep.graph)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        # `dim` is not the second dimension.
        pytest.param((10, 32, 32, 8), 2, id="dim_2"),
        pytest.param((10, 32, 32, 8), -1, id="dim_-1"),
        pytest.param((10, 32, 32, 8), 3, id="dim_3"),
    ],
)
def test_softmax_delegation__unsupported_dims__channels_first(input_shape, dim: int):
    model = SoftmaxModule(dim)
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()
    assert_softmax_not_delegated(delegated_ep.graph)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((4096 + 1, 8), -1, id="2D_WxH_exceeded"),
        pytest.param((4096, 2, 8), -1, id="3D_WxH_exceeded"),
        pytest.param((2, 64, 64, 8), -1, id="4D_WxH_exceeded"),
        pytest.param((1, 2048), -1, id="2D_channels_exceeded"),
        pytest.param((4096, 128 + 8), -1, id="2D_total_size_exceeded"),
        pytest.param((64, 1, 64, 128 + 8), -1, id="4D_total_size_exceeded"),
    ],
)
def test_softmax_delegation__unsupported_dimension_sizes(input_shape, dim: int):
    model = SoftmaxModule(dim)
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()
    assert_softmax_not_delegated(delegated_ep.graph)


@pytest.mark.parametrize(
    "input_shape,dim",
    [
        pytest.param((2, 8, 64, 64), -1, id="4D_WxH_exceeded"),
        pytest.param((64, 128 + 8, 1, 64), -1, id="4D_total_size_exceeded"),
    ],
)
def test_softmax_delegation__unsupported_dimension_sizes__channels_first(
    input_shape, dim: int
):
    model = ConvSoftmaxModule(dim, input_shape[1])
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()
    assert_softmax_not_delegated(delegated_ep.graph)


def test_softmax_delegation__1d():
    input_shape = (8,)
    dim = 0

    model = SoftmaxModule(dim)
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()
    assert_softmax_not_delegated(delegated_ep.graph)
