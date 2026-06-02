# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import (
    ConstantPadNDConvModule,
    ConstantPadNDModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import ConstantPadND, Convolution
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestConstantPadNDNewNeutronFlow:
    """The PyTorch padding is added to the individual dimensions from the back (slightly confusing), see:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker, use_qat=False):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={ConstantPadND: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            use_qat=use_qat,
            use_new_flow_neutron_c=True,
        )

    def assert_delegated_and_output_shape_equals(
        self, model, input_shape, expected_output_shape, mocker
    ):
        model_builder_spy = mocker.spy(ModelBuilder, "finish")

        self.assert_delegated(model, input_shape, mocker)

        neutron_ir_subgraph = model_builder_spy.call_args[0][0].get_sub_graph()
        assert neutron_ir_subgraph.outputs.tmp_outputs[0].shape.vector == list(
            expected_output_shape
        )

    @pytest.mark.parametrize(
        "input_shape, paddings",
        [
            pytest.param((2,), tuple(range(2)), id="1D, padding H"),
            pytest.param((2, 4), tuple(range(2)), id="2D, padding H"),
            pytest.param((2, 4), tuple(range(4)), id="2D, padding N, H"),
            pytest.param((2, 4, 6), tuple(range(2)), id="3D, padding H"),
            pytest.param((2, 4, 6), tuple(range(4)), id="3D, padding C, H"),
            pytest.param((2, 4, 6, 8), tuple(range(2)), id="4D, padding W"),
            pytest.param((2, 4, 6, 8), tuple(range(4)), id="4D, padding H, W"),
            pytest.param((1, 2, 3, 4, 5), tuple(range(2)), id="5D, padding D"),
            pytest.param((1, 2, 3, 4, 5), tuple(range(4)), id="5D, padding W, D"),
        ],
    )
    def test__basic_nsys_inference(self, mocker, input_shape, paddings, use_qat):
        # These test cases are also supported by the old flow.
        model = ConstantPadNDModule(paddings)
        self.assert_delegated(model, input_shape, mocker, use_qat)

    def test__channels_padding(self, mocker):
        input_shape = (2, 4, 6)
        # These paddings will be applied to the last dimension, which is the channels as the input is formatless.
        paddings = (1, 1)
        expected_output_shape = (2, 4, 8)  # Padded channels.
        model = ConstantPadNDModule(paddings)

        self.assert_delegated_and_output_shape_equals(
            model, input_shape, expected_output_shape, mocker
        )

    def test__batch_padding(self, mocker):
        input_shape = (2, 4, 6)
        paddings = (0, 0, 0, 0, 1, 1)  # Padding applied to the batch dimension.
        expected_output_shape = (4, 4, 6)  # Padded batch.
        model = ConstantPadNDModule(paddings)

        self.assert_delegated_and_output_shape_equals(
            model, input_shape, expected_output_shape, mocker
        )

    @pytest.mark.parametrize("constant", [0.0, -13.37])
    def test__specific_constant(self, mocker, constant):
        input_shape = (2, 4, 6)
        paddings = (1, 1)
        model = ConstantPadNDModule(paddings, constant)
        self.assert_delegated(model, input_shape, mocker)

    @pytest.mark.parametrize(
        "input_shape, paddings",
        [
            pytest.param((1, 4, 6, 8), tuple(range(2)), id="4D, padding W"),
            pytest.param((1, 4, 6, 8), tuple(range(4)), id="4D, padding H, W"),
            pytest.param((1, 2, 6, 8), (0, 1, 2, 3, 1, 1), id="4D, padding H, W"),
        ],
    )
    def test__channels_first(self, mocker, input_shape, paddings):
        model = ConstantPadNDConvModule(paddings)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={ConstantPadND: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model, input_shape, graph_verifier, use_new_flow_neutron_c=True
        )
