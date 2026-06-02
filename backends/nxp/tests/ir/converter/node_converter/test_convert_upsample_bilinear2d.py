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
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    UpsampleBilinear2D,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class UpsampleBilinearModule(torch.nn.Module):

    def __init__(self, size=None, scale=None, **kwargs):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            size=size, scale_factor=scale, mode="bilinear", **kwargs
        )

    def forward(self, x):
        return self.upsample(x)


class UpsampleBilinearAddModule(UpsampleBilinearModule):

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


class TestUpsampleBilinear2DNewNeutronFlow:
    # TODO Use quantized dataset and `atol=1` in the tests.

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        use_qat=False,
        atol=None,
        expected_delegated_ops=None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {UpsampleBilinear2D: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        kwargs = {"atol": atol} if atol is not None else {}
        output_comparator = AllCloseOutputComparator(**kwargs)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            output_comparator,
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
        assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleBilinear2D])

    def test__qat__align_corners(self, mocker, use_qat):
        align_corners = True
        input_shape = (1, 2, 3, 4)
        output_size = (5, 7)
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        atol = 0.015  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat, atol=atol)

    def test__qat__not_align_corners(self, mocker, use_qat):
        align_corners = False
        input_shape = (1, 2, 3, 4)
        output_size = (6, 8)
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        atol = 0.015  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat, atol=atol)

    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 2, 3, 4), (6, 8), id="batch=1, scale_h=scale_w=2"),
            pytest.param(
                (3, 3, 3, 5),
                (6, 5),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (3, 16), id="batch=2, scale_h=1, scale_w=4"),
            pytest.param((2, 2, 3, 4), (24, 8), id="batch=2, scale_h=8, scale_w=2"),
        ],
    )
    def test__not_align_corners__output_size(self, mocker, input_shape, output_size):
        align_corners = False
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        atol = 0.016  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    def test__not_align_corners__output_size__unsupported(self):
        align_corners = False
        input_shape = (1, 2, 3, 4)
        output_size = (9, 12)  # scale = (3, 3)
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, scale",
        [
            pytest.param((1, 2, 3, 4), (2, 2), id="batch=1, scale_h=scale_w=2"),
            pytest.param(
                (3, 3, 3, 5),
                (2, 1),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (4, 1), id="batch=2, scale_h=4, scale_w=1"),
            pytest.param((2, 2, 3, 4), (2, 8), id="batch=2, scale_h=2, scale_w=8"),
        ],
    )
    def test__not_align_corners__scales(self, mocker, input_shape, scale):
        align_corners = False
        model = UpsampleBilinearModule(scale=scale, align_corners=align_corners)
        atol = 0.016  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    def test__not_align_corners__scales__unsupported(self):
        align_corners = False
        input_shape = (1, 2, 3, 4)
        scale = (3, 3)
        model = UpsampleBilinearModule(scale=scale, align_corners=align_corners)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 2, 4, 5), (7, 9), id="batch=1, scale_h=scale_w=2"),
            pytest.param(
                (1, 3, 3, 5),
                (5, 5),
                id="batch=1, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 4, 5), (4, 17), id="batch=2, scale_h=1, scale_w=4"),
            pytest.param((1, 2, 4, 5), (25, 9), id="batch=1, scale_h=8, scale_w=2"),
            pytest.param((2, 2, 4, 5), (25, 9), id="batch=2, scale_h=8, scale_w=2"),
            pytest.param(
                (3, 3, 3, 5),
                (5, 5),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
        ],
    )
    def test__align_corners__output_size(self, mocker, input_shape, output_size):
        align_corners = True
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        atol = 0.016  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    def test__align_corners__output_size__unsupported(self):
        align_corners = True
        input_shape = (1, 2, 3, 4)
        output_size = (6, 8)  # Neutron scale = (5/2, 7/3)
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        self.assert_not_delegated(model, input_shape)

    def test__align_corners__output_size__input_size_equal_to_one(self):
        align_corners = True
        input_shape = (1, 2, 1, 1)  # Neutron scale computation would divide by zero.
        output_size = (2, 2)
        model = UpsampleBilinearModule(size=output_size, align_corners=align_corners)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, scale",
        [
            # The PyTorch scales are "weird" because the "Neutron scales" are computed differently.
            # The fractions correspond to "nice" Neutron scales (1, 2, 4, or 8).
            pytest.param(
                (1, 2, 4, 5),
                (7 / 4, 9 / 5),
                id="batch=1, scale_h=7/4, scale_w=9/5 (Neutron scales = (2, 2)",
            ),
            pytest.param(
                (1, 3, 3, 5),
                (5 / 3, 1),
                id="batch=1, scale_h=5/3, scale_w=1 (Neutron scales = (2, 1))",
            ),
            pytest.param(
                (2, 2, 4, 5),
                (1, 17 / 5),
                id="batch=2, scale_h=1, scale_w=17/5 (Neutron scales = (1, 4))",
            ),
            pytest.param(
                (1, 2, 4, 5),
                (25 / 4, 9 / 5),
                id="batch=1, scale_h=25/4, scale_w=9/5 (Neutron scales = (8, 2))",
            ),
            pytest.param(
                (2, 2, 4, 5),
                (25 / 4, 9 / 5),
                id="batch=3, scale_h=25/4, scale_w=9/5 (Neutron scales = (8, 2))",
            ),
            pytest.param(
                (3, 3, 3, 5),
                (5 / 3, 1),
                id="batch=3, scale_h=5/3, scale_w=1 (Neutron scales = (2, 1))",
            ),
        ],
    )
    def test__align_corners__scales(self, mocker, input_shape, scale):
        align_corners = True
        model = UpsampleBilinearModule(scale=scale, align_corners=align_corners)
        atol = 0.016  # ~= output scale -> single bit error.
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    def test__align_corners__scales__unsupported(self):
        align_corners = True
        input_shape = (1, 2, 3, 4)
        scale = (2, 2)  # Neutron scale = (5/2, 7/3)
        model = UpsampleBilinearModule(scale=scale, align_corners=align_corners)
        self.assert_not_delegated(model, input_shape)

    def test__noop__alone_in_partition__not_delegated(self):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleBilinearModule(scale=scale)
        self.assert_not_delegated(model, input_shape)

    def test__noop__not_alone_in_partition__delegated(self, mocker):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleBilinearAddModule(scale=scale)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={UpsampleBilinear2D: 1, AddTensor: 1},
        )
