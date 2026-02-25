# Copyright 2024,2026 NXP
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
from executorch.backends.nxp.tests.use_qat import *  # noqa F403

# noinspection PyProtectedMember
from executorch.exir.dialects._ops import ops as exir_ops

ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
MaxPool2D = exir_ops.edge.aten.max_pool2d.default


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, **kwargs)

    def forward(self, x):
        return self.max_pool2d(x)


def _generate_test_data(input_shape: tuple) -> np.ndarray:
    """Generate random int8 test data for given shape."""
    return (np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0).astype(
        np.int8
    )


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestMaxPool2DSupported:
    """Tests for supported MaxPool2D configurations."""

    @staticmethod
    def _verify_successful_delegation(module, converter_spy, input_shape):
        edge_model = to_quantized_edge_program(
            module,
            input_shape,
            use_neutron_for_format_conversion=False,
        ).exported_program()

        # Make sure the MaxPool was delegated.
        assert not graph_contains_any_of_ops(edge_model.graph, [MaxPool2D])
        assert graph_contains_any_of_ops(edge_model.graph, [ExecutorchDelegateCall])

        # Verify correct behavior of the converted NeutronIR model.
        edge_partition = converter_spy.call_args.args[1]
        neutron_ir_partition, _ = converter_spy.spy_return

        input_data = _generate_test_data(input_shape)

        # Make sure the tested program contains the `MaxPool`.
        assert graph_contains_any_of_ops(edge_partition.graph, [MaxPool2D])

        convert_run_compare(
            edge_partition,
            tfl_model=neutron_ir_partition,
            input_data=input_data,
            tflite_input_preprocess=ToChannelLastPreprocess(),
            tflite_output_preprocess=ToChannelFirstPreprocess(),
        )

    @pytest.mark.parametrize(
        "padding",
        [(0, 0), (1, 1), (0, 1), 0, 1],
        ids=lambda padding: f"Padding = {'tuple' if isinstance(padding, tuple) else 'scalar'} `{padding}`",
    )
    def test_padding(self, padding, mocker):
        input_shape = (1, 8, 5, 6)
        stride = 1  # Default value would be equal to kernel size (3), which is not supported by Neutron.
        module = MaxPool2dModule(kernel_size=3, stride=stride, padding=padding)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
        self._verify_successful_delegation(module, converter_spy, input_shape)

    @pytest.mark.parametrize(
        "stride",
        [(1, 1), (2, 1), (2, 2), (2, 3), (2, 8), 1, 2],
        ids=lambda stride: f"Stride = {'tuple' if isinstance(stride, tuple) else 'scalar'} `{stride}`",
    )
    def test_stride(self, stride, mocker):
        input_shape = (1, 8, 7, 9)
        module = MaxPool2dModule(kernel_size=3, stride=stride)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
        self._verify_successful_delegation(module, converter_spy, input_shape)


class TestMaxPool2DUnsupported:
    """Tests for unsupported MaxPool2D configurations."""

    @staticmethod
    def _verify_no_delegation(module, input_shape):
        edge_model = to_quantized_edge_program(
            module,
            input_shape,
            use_neutron_for_format_conversion=False,
        ).exported_program()

        assert graph_contains_any_of_ops(edge_model.graph, [MaxPool2D])
        assert not graph_contains_any_of_ops(edge_model.graph, [ExecutorchDelegateCall])

    def test_unsupported_dilation(self):
        dilation = 2  # Unsupported.
        input_shape = (1, 8, 7, 9)

        module = MaxPool2dModule(kernel_size=3, dilation=dilation)

        # Make sure the MaxPool was NOT delegated.
        self._verify_no_delegation(module, input_shape)

    def test_unsupported_stride(self):
        stride = 3  # Unsupported.
        input_shape = (1, 8, 7, 9)

        module = MaxPool2dModule(kernel_size=3, stride=stride)

        # Make sure the MaxPool was NOT delegated.
        self._verify_no_delegation(module, input_shape)

    def test_unsupported_padding(self):
        padding = 4  # Unsupported. Bigger than kernel size.
        input_shape = (1, 8, 7, 9)

        with pytest.raises(
            RuntimeError, match=r"pad should be at most half of effective kernel size"
        ):
            to_quantized_edge_program(
                MaxPool2dModule(kernel_size=3, padding=padding),
                input_shape,
                use_neutron_for_format_conversion=False,
            ).exported_program()

    def test_unsupported_ceil_mode(self):
        ceil_mode = True  # Unsupported.
        input_shape = (1, 8, 7, 9)

        module = MaxPool2dModule(kernel_size=3, ceil_mode=ceil_mode)

        # Make sure the MaxPool was NOT delegated.
        self._verify_no_delegation(module, input_shape)

    def test_unsupported_batch_size(self):
        batch_size = 2  # Unsupported.
        input_shape = (batch_size, 8, 7, 9)

        module = MaxPool2dModule(kernel_size=3)

        # Make sure the MaxPool was NOT delegated.
        self._verify_no_delegation(module, input_shape)

    def test_unsupported_channels(self):
        channels = 3  # Unsupported. Must be a multiple of `num_macs` (`8`).
        input_shape = (1, channels, 7, 9)

        module = MaxPool2dModule(kernel_size=3)

        # Make sure the MaxPool was NOT delegated.
        self._verify_no_delegation(module, input_shape)
