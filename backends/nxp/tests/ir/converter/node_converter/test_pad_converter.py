# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import PadConvModule, PadModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Convolution, Pad


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestPadConverter:
    """The PyTorch padding is added to the individual dimensions from the back (slightly confusing), see:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad

    Current `pad_converter` currently converts only padding with mode `reflect`. Mode `constant` padding
    is decomposed into `constant_pad_nd` and has its own converter.

    Padding with `reflect` mode has the following constraints:
    - len(padding) = 2 ... only 2D/3D input
    - len(padding) = 4 ... only 3D/4D input
    - len(padding) = 6 ... only 4D/5D input
    - len(padding) = 8 and higher not implemented in Torch

    Thus padding the first dim directly (without permute) is not possible.
    Mainly, this constraint disables padding of the outer-most dimension (ie. batch).
    """

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self, model, input_shape, mocker, request, exp_deleg_nodes=None, use_qat=False
    ):
        if exp_deleg_nodes is None:
            exp_deleg_nodes = {Pad: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=exp_deleg_nodes,
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            use_qat=use_qat,
        )

    def assert_delegated_and_output_shape_equals(
        self, model, input_shape, expected_output_shape, mocker, request
    ):
        model_builder_spy = mocker.spy(ModelBuilder, "finish")

        self.assert_delegated(model, input_shape, mocker, request)

        neutron_ir_subgraph = model_builder_spy.call_args[0][0].get_sub_graph()
        assert neutron_ir_subgraph.outputs.tmp_outputs[0].shape.vector == list(
            expected_output_shape
        )

    @pytest.mark.parametrize(
        "input_shape, paddings",
        [
            pytest.param((3, 5), tuple(range(2)), id="2D, padding one dim"),
            pytest.param((3, 3, 5), tuple(range(2)), id="3D, padding one dim"),
            pytest.param((3, 7, 5), tuple(range(4)), id="3D, padding two dims"),
            pytest.param((3, 3, 7, 5), tuple(range(4)), id="4D, padding two dims"),
            pytest.param((3, 9, 7, 5), tuple(range(6)), id="4D, padding three dims"),
            pytest.param((3, 3, 9, 7, 5), tuple(range(6)), id="5D, padding three dims"),
        ],
    )
    def test__basic__reflect(self, mocker, request, input_shape, paddings):
        model = PadModule(paddings=paddings, mode="reflect")

        self.assert_delegated(model, input_shape, mocker, request)

    def test__channels_padding__reflect(self, mocker, request):
        input_shape = (2, 4, 6)
        # These paddings will be applied to the last dimension, which is the channels as the input is formatless.
        paddings = (1, 1)
        expected_output_shape = (2, 4, 8)  # Padded channels.
        model = PadModule(paddings, "reflect")

        self.assert_delegated_and_output_shape_equals(
            model, input_shape, expected_output_shape, mocker, request
        )

    @pytest.mark.parametrize(
        "input_shape, paddings",
        [
            pytest.param((1, 10, 8, 6), tuple(range(4)), id="4D, padding H, W"),
            pytest.param((1, 10, 8, 6), tuple(range(6)), id="4D, padding C, H, W"),
        ],
    )
    def test__channels_first__reflect(self, mocker, request, input_shape, paddings):
        # compute channels size after padding
        if len(paddings) // 2 == len(input_shape) - 1:
            conv_in_channels = input_shape[1] + paddings[-1] + paddings[-2]
        else:
            conv_in_channels = input_shape[1]
        model = PadConvModule(conv_in_channels, paddings, "reflect")

        self.assert_delegated(
            model, input_shape, mocker, request, {Pad: 1, Convolution: 1}
        )

    def test__qat__reflect(self, mocker, request):
        input_shape = (2, 4, 6)
        paddings = (1, 1)
        model = PadModule(paddings, "reflect")

        self.assert_delegated(model, input_shape, mocker, request, use_qat=True)
