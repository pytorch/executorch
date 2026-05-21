# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.constant_pad_nd_converter import (
    ConstantPadNDConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_edge_program,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    OverrideTargetSupportCheck,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
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


@pytest.mark.parametrize("constant", [0.0, 42.0, -13.37])
def test_constant_pad_nd_conversion__specific_constant(constant):
    input_shape = (2, 4, 6, 8)
    paddings = (1, 2, 3, 4)

    edge_program = to_edge_program(
        ConstantPadNDModule(paddings, constant), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    # Ignore the target requirement, as this test is target agnostic.
    def supported_target(*_):
        return True

    with OverrideTargetSupportCheck(
        ConstantPadNDConverter, new_target_support_check=supported_target
    ):
        convert_run_compare(edge_program, input_data)


def test_constant_pad_nd_conversion__default_constant():
    input_shape = (2, 4, 6, 8)
    paddings = (1, 2, 3, 4)

    edge_program = to_edge_program(
        ConstantPadNDModule(paddings), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    # Ignore the target requirement, as this test is target agnostic.
    def supported_target(*_):
        return True

    with OverrideTargetSupportCheck(
        ConstantPadNDConverter, new_target_support_check=supported_target
    ):
        convert_run_compare(edge_program, input_data)


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
def test_constant_pad_nd_conversion__format_less(input_shape, paddings):
    edge_program = to_edge_program(
        ConstantPadNDModule(paddings), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    # Ignore the target requirement, as this test is target agnostic.
    def supported_target(*_):
        return True

    with OverrideTargetSupportCheck(
        ConstantPadNDConverter, new_target_support_check=supported_target
    ):
        convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize(
    "input_shape, paddings",
    [
        pytest.param((1, 4, 6, 8), tuple(range(2)), id="4D, padding W"),
        pytest.param((1, 4, 6, 8), tuple(range(4)), id="4D, padding H, W"),
    ],
)
def test_constant_pad_nd_conversion__channels_first(input_shape, paddings):
    model = ConstantPadNDConvModule(paddings)
    edge_program = to_edge_program(
        model, input_shape
    ).exported_program()  # Extra `Conv` after the padding.

    input_data = np.random.random(input_shape).astype(np.float32)

    # Ignore the target requirement, as this test is target agnostic.
    def supported_target(*_):
        return True

    with OverrideTargetSupportCheck(
        ConstantPadNDConverter, new_target_support_check=supported_target
    ):
        convert_run_compare(
            edge_program,
            input_data,
            tflite_input_preprocess=ToNHWCPreprocess(),
            tflite_output_preprocess=ToNCHWPreprocess(),
            conversion_config=ConversionConfig(
                {"use_neutron_for_format_conversion": False}
            ),
        )


@pytest.mark.parametrize(
    "input_shape, paddings",
    [
        pytest.param((2, 4, 6), tuple(range(6)), id="3D, padding N, C, H"),
        pytest.param((2, 4, 6, 8), tuple(range(6)), id="4D, padding C, H, W"),
        pytest.param((2, 4, 6, 8), tuple(range(8)), id="4D, padding N, C, H, W"),
        pytest.param((1, 2, 3, 4, 5), tuple(range(6)), id="5D, padding H, W, D"),
        pytest.param((1, 2, 3, 4, 5), tuple(range(8)), id="5D, padding C, H, W, D"),
        pytest.param((1, 2, 3, 4, 5), tuple(range(10)), id="5D, padding N, C, H, W, D"),
        pytest.param((1, 1, 6, 8), (1, 2, 3, 4, 2, 1), id="4D, padding C, H, W"),
    ],
)
def test_constant_pad_nd__unsupported_paddings(input_shape, paddings, use_qat):
    model = ConstantPadNDModule(paddings)
    exec_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # There is at least one non-delegated Pad node
    assert graph_contains_any_of_ops(exec_program.graph, [ConstantPadND])


def test_constant_pad_nd__delegation__formatless__supported_padding(use_qat):
    input_shape = (2, 4, 6, 8)  # Formatless -> the last dim (8) will be padded.
    paddings = [0, 0, 1, 2, 3, 4]  # The last dim is padded using the first 2 paddings.
    model = ConstantPadNDModule(paddings)
    exec_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_new_flow_neutron_c=True
    ).exported_program()

    # Make sure the `pad` was delegated.
    assert not graph_contains_any_of_ops(exec_program.graph, [ConstantPadND])


def test_constant_pad_nd__delegation__formatless__unsupported_padding(use_qat):
    input_shape = (2, 4, 6, 8)  # Formatless -> the last dim (8) will be padded.
    paddings = [0, 1]  # The last dim is padded using the first 2 paddings.
    model = ConstantPadNDModule(paddings)
    exec_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `pad` was NOT delegated.
    assert graph_contains_any_of_ops(exec_program.graph, [ConstantPadND])


def test_constant_pad_nd__delegation__channels_first__supported_padding(use_qat):
    input_shape = (2, 4, 6, 8)  # Channels first -> the second dim (4) will be padded.
    paddings = [1, 2, 3, 4, 0, 0]  # The second dim is padded using the paddings[4:6].
    model = ConstantPadNDConvModule(paddings)
    exec_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_new_flow_neutron_c=True
    ).exported_program()

    # Make sure the `pad` was delegated.
    assert not graph_contains_any_of_ops(exec_program.graph, [ConstantPadND])


def test_constant_pad_nd__delegation__channels_first__unsupported_padding(use_qat):
    input_shape = (2, 3, 6, 8)  # Channels first -> the second dim (3) will be padded.
    paddings = [0, 0, 0, 0, 1, 0]  # The second dim is padded using the paddings[4:6].
    model = ConstantPadNDConvModule(paddings)
    exec_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `pad` was NOT delegated.
    assert graph_contains_any_of_ops(exec_program.graph, [ConstantPadND])


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

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Known issue in Neutron: https://jira.sw.nxp.com/browse/AIR-14624",  # @lint-ignore
    )
    def test__bugged_channels_first_case(self, mocker):
        input_shape, paddings = (1, 2, 6, 8), (0, 1, 2, 3, 1, 1)
        model = ConstantPadNDConvModule(paddings)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={ConstantPadND: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model, input_shape, graph_verifier, use_new_flow_neutron_c=True
        )
