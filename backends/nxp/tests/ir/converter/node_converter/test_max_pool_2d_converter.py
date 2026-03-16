# Copyright 2024,2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

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
GetItem = operator.getitem
MaxPool2D = exir_ops.edge.aten.max_pool2d_with_indices.default
Squeeze = exir_ops.edge.aten.squeeze.default
SqueezeDim = exir_ops.edge.aten.squeeze.dim
SqueezeDims = exir_ops.edge.aten.squeeze.dims
Unsqueeze = exir_ops.edge.aten.unsqueeze.default
ViewCopy = exir_ops.edge.aten.view_copy.default


class MaxPool1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
        )

    def forward(self, x):
        return self.max_pool(x)


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
        assert graph_contains_any_of_ops(edge_partition.graph, [GetItem])

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
        assert graph_contains_any_of_ops(edge_model.graph, [GetItem])
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


class TestMaxPool1D:
    """There is no `max_pool1d` in the edge dialect. During lowering to edge, ExecuTorch extends the shape to 4D (with
    a `1`), then applies `max_pool2d`, and then removes the `1` from the shape to make it 3D again. So the aten
    `max_pool1d` is handled by the `max_pool2d` support. This test verifies that the lowering process works correctly.
    """

    def test_max_pool_2d__from_1d(self, mocker):
        model = MaxPool1DModule()
        input_shape = (1, 8, 12)
        extended_shape = (1, 8, 1, 12)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
        edge_model = to_quantized_edge_program(
            model, input_shape, use_neutron_for_format_conversion=False
        ).exported_program()

        # Make sure the `max_pool` was delegated.
        assert graph_contains_any_of_ops(edge_model.graph, [ExecutorchDelegateCall])
        assert not graph_contains_any_of_ops(edge_model.graph, [MaxPool2D])
        # There is not `max_pool1d` in the edge dialect, so we cannot check for its absence by comparing with the target.
        # In order to detect any potential future changes (like the addition of `max_pool1d` to edge dialect), we check
        #  the name of the target.
        assert not any(
            n for n in edge_model.graph.nodes if "1d" in str(n.target)
        )  # Check for anything 1D.

        # Make sure both `view_copy` nodes were added, and there is no `squeeze` or `unsqueeze`.
        assert len([n for n in edge_model.graph.nodes if n.target == ViewCopy]) == 2
        assert not graph_contains_any_of_ops(
            edge_model.graph, [Unsqueeze, Squeeze, SqueezeDim, SqueezeDims]
        )

        # Verify correct behavior of the converted NeutronIR model.
        edge_partition = converter_spy.call_args.args[1]
        neutron_ir_partition, _ = converter_spy.spy_return

        input_data = _generate_test_data(extended_shape)

        # Make sure the tested program contains the `MaxPool`.
        assert graph_contains_any_of_ops(edge_partition.graph, [MaxPool2D])
        assert graph_contains_any_of_ops(edge_partition.graph, [GetItem])

        convert_run_compare(
            edge_partition,
            tfl_model=neutron_ir_partition,
            input_data=input_data,
            tflite_input_preprocess=ToChannelLastPreprocess(),
            tflite_output_preprocess=ToChannelFirstPreprocess(),
        )
