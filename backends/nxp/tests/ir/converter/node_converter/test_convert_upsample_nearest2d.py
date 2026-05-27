# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    UpsampleNearest2D,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class UpsampleNearestModule(torch.nn.Module):

    def __init__(self, size=None, scale=None):
        super().__init__()
        self.upsample = torch.nn.Upsample(size=size, scale_factor=scale, mode="nearest")

    def forward(self, x):
        return self.upsample(x)


class UpsampleNearestAddModule(UpsampleNearestModule):

    def forward(self, x):
        x = super().forward(x)
        return x + x


@pytest.mark.parametrize(
    "input_shape, size",
    [
        pytest.param((1, 8, 2, 3), (4, 6), id="2x upscale, 8 channels, tuple size"),
        pytest.param((1, 8, 3, 3), 6, id="2x upscale, 8 channels, scalar size"),
        pytest.param((1, 8, 2, 3), (8, 12), id="4x upscale, 8 channels, tuple size"),
        pytest.param((1, 8, 3, 3), 12, id="4x upscale, 8 channels, scalar size"),
    ],
)
@pytest.mark.xfail(strict=True, reason="EIEX-881")
def test_convert_upsample_nearest2d__size(mocker, input_shape, size):
    model = UpsampleNearestModule(size=size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `upsample`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [UpsampleNearest2D])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
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
@pytest.mark.xfail(strict=True, reason="EIEX-881")
def test_convert_upsample_nearest2d__scale_factor(mocker, input_shape, scale_factor):
    model = UpsampleNearestModule(scale=scale_factor)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `upsample`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [UpsampleNearest2D])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


def test_convert_upsample_nearest2d__no_delegation__unsupported_channels():
    size = 6
    input_shape = (1, 2, size // 2, size // 2)  # 2 channels, not `num_macs`.
    model = UpsampleNearestModule(size=size)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (channels != 8).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        pytest.param((1, 8, 4, 4), 3, id="3x upscale"),
        pytest.param((1, 8, 4, 4), 1.5, id="1.5x upscale"),
        pytest.param((1, 8, 4, 4), (2, 4), id="2x and 4x mixed upscale"),
        pytest.param((1, 8, 10, 10), 1.99, id="1.99x upscale"),
    ],
)
def test_convert_upsample_nearest2d__no_delegation__unsupported_scale(
    input_shape, scale_factor
):
    model = UpsampleNearestModule(scale=scale_factor)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (scale != 2).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])


@pytest.mark.parametrize(
    "input_shape, size",
    [
        pytest.param((1, 8, 2, 3), (6, 9), id="3x upscale"),
        pytest.param((1, 8, 2, 4), (3, 6), id="1.5x upscale"),
        pytest.param((1, 8, 3, 4), 6, id="non-uniform upscale"),
    ],
)
def test_convert_upsample_nearest2d__no_delegation__unsupported_size(input_shape, size):
    model = UpsampleNearestModule(size=size)

    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `upsample` was NOT delegated (size != double of input).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])


class TestUpsampleNearest2DNewNeutronFlow:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        use_qat=False,
        expected_delegated_ops=None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {UpsampleNearest2D: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_qat=use_qat,
            use_new_flow_neutron_c=True,  # Use the new flow.
        )

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(
            model, input_shape, use_new_flow_neutron_c=True
        ).exported_program()

        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])

    def test__qat(self, mocker, use_qat):
        input_shape = (1, 2, 3, 4)
        output_size = (6, 8)
        model = UpsampleNearestModule(size=output_size)
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 2, 3, 4), (6, 8), id="batch=1, scale_h=scale_w=2"),
            pytest.param((1, 2, 3, 3), 6, id="batch=1, scale_h=scale_w=2, scalar size"),
            pytest.param(
                (3, 3, 3, 5),
                (6, 5),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (3, 16), id="batch=2, scale_h=1, scale_w=4"),
            pytest.param((2, 2, 3, 4), (24, 8), id="batch=2, scale_h=8, scale_w=2"),
        ],
    )
    def test__output_size(self, mocker, input_shape, output_size):
        model = UpsampleNearestModule(size=output_size)
        self.assert_delegated(model, input_shape, mocker)

    def test__output_size__unsupported(self):
        input_shape = (1, 2, 3, 4)
        output_size = (9, 12)  # scale = (3, 3)
        model = UpsampleNearestModule(size=output_size)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, scale",
        [
            pytest.param((1, 2, 3, 4), (2, 2), id="batch=1, scale_h=scale_w=2"),
            pytest.param(
                (1, 2, 3, 4), 4, id="batch=1, scale_h=scale_w=4, scalar scale"
            ),
            pytest.param(
                (3, 3, 3, 5),
                (2, 1),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (4, 1), id="batch=2, scale_h=4, scale_w=1"),
            pytest.param((2, 2, 3, 4), (2, 8), id="batch=2, scale_h=2, scale_w=8"),
        ],
    )
    def test__scales(self, mocker, input_shape, scale):
        model = UpsampleNearestModule(scale=scale)
        self.assert_delegated(model, input_shape, mocker)

    def test__scales__unsupported(self):
        input_shape = (1, 2, 3, 4)
        scale = (3, 3)
        model = UpsampleNearestModule(scale=scale)
        self.assert_not_delegated(model, input_shape)

    def test__noop__alone_in_partition__not_delegated(self):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleNearestModule(scale=scale)
        self.assert_not_delegated(model, input_shape)

    def test__noop__not_alone_in_partition__delegated(self, mocker):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleNearestAddModule(scale=scale)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={UpsampleNearest2D: 1, AddTensor: 1},
        )
