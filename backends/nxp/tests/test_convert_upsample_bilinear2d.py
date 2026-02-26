# Copyright 2026 NXP
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
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


# noinspection PyProtectedMember
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
UpsampleBilinear2D = exir_ops.edge.aten.upsample_bilinear2d.vec


class UpsampleBilinearModule(torch.nn.Module):

    def __init__(self, size=None, scale=None):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            size=size, scale_factor=scale, mode="bilinear"
        )

    def forward(self, x):
        return self.upsample(x)


@pytest.mark.parametrize(
    "input_shape, size",
    [
        pytest.param((1, 8, 2, 3), (4, 6), id="2x upscale, 8 channels, tuple size"),
        pytest.param((1, 8, 3, 3), 6, id="2x upscale, 8 channels, scalar size"),
        pytest.param((1, 8, 2, 3), (8, 12), id="4x upscale, 8 channels, tuple size"),
        pytest.param((1, 8, 3, 3), 12, id="4x upscale, 8 channels, scalar size"),
    ],
)
def test_convert_upsample_bilinear2d__size(mocker, input_shape, size):
    model = UpsampleBilinearModule(size=size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `upsample`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [UpsampleBilinear2D])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=1,  # Common quantized rounding error.
    )


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        pytest.param((1, 8, 2, 3), 2, id="2x upscale, 8 channels, scalar scale"),
        pytest.param((1, 8, 3, 3), 2.0, id="2x upscale, 8 channels, float scale"),
        pytest.param((1, 8, 4, 5), (2, 2), id="2x upscale, 8 channels, tuple scale"),
        pytest.param((1, 8, 2, 3), 4, id="4x upscale, 8 channels, scalar scale"),
        pytest.param((1, 8, 2, 3), (4, 4), id="4x upscale, 8 channels, tuple scale"),
    ],
)
def test_convert_upsample_bilinear2d__scale_factor(mocker, input_shape, scale_factor):
    model = UpsampleBilinearModule(scale=scale_factor)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `upsample`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [UpsampleBilinear2D])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=1,  # Common quantized rounding error.
    )


def test_convert_upsample_bilinear2d__no_delegation__unsupported_channels():
    size = 6
    input_shape = (1, 2, size // 2, size // 2)  # 2 channels, not `num_macs`.
    model = UpsampleBilinearModule(size=size)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (channels != 8).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        pytest.param((1, 8, 4, 4), 3, id="3x upscale"),
        pytest.param((1, 8, 4, 4), 1.5, id="1.5x upscale"),
        pytest.param((1, 8, 4, 4), (2, 4), id="2x and 4x mixed upscale"),
        pytest.param((1, 8, 10, 10), 1.99, id="1.99x upscale"),
    ],
)
def test_convert_upsample_bilinear2d__no_delegation__unsupported_scale(
    input_shape, scale_factor
):
    model = UpsampleBilinearModule(scale=scale_factor)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (scale != 2).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])


@pytest.mark.parametrize(
    "input_shape, size",
    [
        pytest.param((1, 8, 2, 3), (6, 9), id="3x upscale"),
        pytest.param((1, 8, 2, 4), (3, 6), id="1.5x upscale"),
        pytest.param((1, 8, 3, 4), 6, id="non-uniform upscale"),
    ],
)
def test_convert_upsample_bilinear2d__no_delegation__unsupported_size(
    input_shape, size
):
    model = UpsampleBilinearModule(size=size)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (size != double of input).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])
