# Copyright 2024,2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
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
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
    Squeeze,
    SqueezeDim,
    SqueezeDims,
    Unsqueeze,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
import pytest


class MaxPool1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
        )

    def forward(self, x):
        return self.max_pool(x)


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size: int | tuple[int, ...] = 3, **kwargs):
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
        assert not graph_contains_any_of_ops(edge_model.graph, [MaxPool2DWithIndices])
        assert graph_contains_any_of_ops(edge_model.graph, [ExecutorchDelegateCall])

        # Verify correct behavior of the converted NeutronIR model.
        edge_partition = converter_spy.call_args.args[1]
        neutron_ir_partition, _ = converter_spy.spy_return

        input_data = _generate_test_data(input_shape)

        # Make sure the tested program contains the `MaxPool`.
        assert graph_contains_any_of_ops(edge_partition.graph, [MaxPool2DWithIndices])
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

        assert graph_contains_any_of_ops(edge_model.graph, [MaxPool2DWithIndices])
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
        assert not graph_contains_any_of_ops(edge_model.graph, [MaxPool2DWithIndices])
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
        assert graph_contains_any_of_ops(edge_partition.graph, [MaxPool2DWithIndices])
        assert graph_contains_any_of_ops(edge_partition.graph, [GetItem])

        convert_run_compare(
            edge_partition,
            tfl_model=neutron_ir_partition,
            input_data=input_data,
            tflite_input_preprocess=ToChannelLastPreprocess(),
            tflite_output_preprocess=ToChannelFirstPreprocess(),
        )


class TestMaxPool2DNewNeutronFlow:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model, input_shape, graph_verifier, use_new_flow_neutron_c=True
        )

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(
            model, input_shape, use_new_flow_neutron_c=True
        ).exported_program()

        # Make sure the `max_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [MaxPool2DWithIndices])

    def test__basic_nsys_inference(self, mocker):
        input_shape = (2, 4, 6, 7)  # The old flow limited the batch size to 1.
        model = MaxPool2dModule()
        self.assert_delegated(model, input_shape, mocker)

    def test__kernel_size_limit(self, mocker):
        kernel_size = (1, 4096)
        input_shape = (1, 4) + kernel_size
        model = MaxPool2dModule(kernel_size)
        self.assert_delegated(model, input_shape, mocker)

    def test__kernel_size_limit_exceeded(self):
        kernel_size = (1, 4097)  # Exceeds the kernel size limit.
        input_shape = (1, 4) + kernel_size
        model = MaxPool2dModule(kernel_size)
        self.assert_not_delegated(model, input_shape)

    def test__stride_limit__no_padding(self, mocker):
        stride = 4096
        input_shape = (1, 4, 1, 4096)
        model = MaxPool2dModule(1, stride=stride)
        self.assert_delegated(model, input_shape, mocker)

    def test__stride_limit_exceeded__no_padding(self):
        stride = 4097  # Exceeds the stride limit.
        input_shape = (1, 4, 1, 4096)
        model = MaxPool2dModule(1, stride=stride)
        self.assert_not_delegated(model, input_shape)

    def test__stride_limit__padding(self, mocker):
        padding = 1
        stride = 4096
        input_shape = (1, 2, 3, stride)
        model = MaxPool2dModule(3, stride=stride, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__stride_limit_exceeded__padding(self):
        padding = 1
        stride = 4097  # Exceeds the stride limit.
        input_shape = (1, 2, 3, stride)
        model = MaxPool2dModule(3, stride=stride, padding=padding)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.skip(
        reason="Large padding requires large kernel size which results in an extremely slow test."
    )
    def test__padding_limit(self, mocker):
        # As the padding is added wia a `Pad` operator (not the `MaxPool` arguments), there is no limit to the padded
        #  value. But as padding can be at most half of the kernel size (PyTorch requirement) and kernel size is limited
        #  to 4096, padding of 2048 is the limit.
        padding = 2048
        kernel_size = padding * 2
        input_shape = (1, 1, 2, 3)
        model = MaxPool2dModule(kernel_size, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__padding__max_pool_limit_exceeded(self, mocker):
        # NeutronIR `MaxPool` padding is limited to 32. But as it is added by the `Pad` operator instead, there is no
        #  limit. This tests ensures the `MaxPool` padding limit is not a problem.
        padding = 33
        kernel_size = padding * 2
        input_shape = (1, 2, 3, 4)
        model = MaxPool2dModule(kernel_size, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__padding_to_kernel_ratio_exceeded(self):
        # Both PyTorch and Neutron require the padding to be at most half of the kernel size.
        kernel_size = 3
        padding = 2  # More than half of the kernel size.
        input_shape = (1, 2, 3, 4)
        model = MaxPool2dModule(kernel_size, padding=padding)
        with pytest.raises(
            RuntimeError, match="pad should be at most half of effective kernel size"
        ):
            to_quantized_edge_program(model, input_shape, use_new_flow_neutron_c=True)


class TestMaxPool1DNewNeutronFlow:

    # Just a basic test to verify that the operator gets extended to the 2D variant correctly.
    def test__basic_nsys_inference__view_not_delegated(self, mocker):
        input_shape = (2, 4, 6)  # The old flow limited the batch size to 1.
        model = MaxPool1DModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={ViewCopy: 2},
        )

        lower_run_compare(
            model, input_shape, graph_verifier, use_new_flow_neutron_c=True
        )
