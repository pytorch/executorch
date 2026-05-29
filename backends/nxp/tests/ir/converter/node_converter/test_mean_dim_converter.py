# Copyright 2025-2026 NXP
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
from executorch.backends.nxp.tests.models import MeanDimConvModule, MeanDimLinearModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
    MeanDim,
)
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class MeanDimModule(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MeanDimAddModule(MeanDimModule):
    def forward(self, x):
        x = super().forward(x)
        return x + x


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 4, 8, 8), (-1, -2), id="Dim -1, -2."),
        pytest.param((1, 4, 8, 8), (-2, -1), id="Dim -2, -1."),
        pytest.param((1, 4, 8, 8), (2, 3), id="Dim 2, 3."),
        pytest.param((1, 4, 8, 8), (3, 2), id="Dim 3, 2."),
    ],
)
def test_mean_dim_conv_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimConvModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
    # Make sure the `mean.dim` was delegated.
    assert not graph_contains_any_of_ops(ep.graph, [MeanDim])
    assert any("lowered_module" in n.name for n in ep.graph.nodes)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        input_data=input_data,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 32), 0, id="Dim 0."),
        pytest.param((1, 32), 1, id="Dim 1."),
    ],
)
@pytest.mark.parametrize(
    "keepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_linear_unsupported_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim
):
    model = MeanDimLinearModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Last 2 dimensions are not used or keepdim is False, cannot be converted to MeanDim, node is not delegated
    assert nodes[6].target == MeanDim

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 4, 8, 8), 0, id="Dim 0."),
        pytest.param((1, 4, 8, 8), 2, id="Dim 2."),
        pytest.param((1, 4, 8, 8), -1, id="Dim -1."),
        pytest.param((1, 4, 8, 8), -2, id="Dim -2."),
        pytest.param((1, 4, 8, 8), (0, 1), id="Dim 0, 1."),
        pytest.param((1, 4, 8, 8), (1, 3), id="Dim 1, 3."),
        pytest.param((1, 4, 8, 8), (-1, -3), id="Dim -1, -3."),
    ],
)
@pytest.mark.parametrize(
    "keepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_conv_unsupported_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim
):
    model = MeanDimConvModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Last 2 dimensions are not used or keepdim is False, cannot be converted to MeanDim, node is not delegated
    assert nodes[6].target == MeanDim

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        input_data=input_data,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        tfl_model=tflite_flatbuffers_model,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 2, 3, 8), (1, 2), id="Dim 1, 2."),
        pytest.param((1, 2, 3, 8), (2, 1), id="Dim 2, 1."),
        pytest.param((1, 2, 3, 8), (-3, -2), id="Dim -3, -2."),
        pytest.param((1, 2, 3, 8), (-2, -3), id="Dim -2, -3."),
    ],
)
def test_mean_dim__formatless__supported(
    mocker, input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was delegated.
    assert not graph_contains_any_of_ops(ep.graph, [MeanDim])
    assert any("lowered_module" in n.name for n in ep.graph.nodes)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 2, 3, 8), (2, 3), id="Dim 2, 3."),
    ],
)
def test_mean_dim__formatless__unsupported(input_shape, dim, use_qat, keepdim=True):
    model = MeanDimModule(dim, keepdim)

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [MeanDim])
    assert not any("lowered_module" in n.name for n in ep.graph.nodes)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 8, 8, 4), (1, 2), id="Dim 1, 2 (supported), channels = 4 (unsupported)."
        ),
    ],
)
def test_mean_dim__formatless__unsupported_channels(
    input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimModule(dim, keepdim)

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [MeanDim])
    assert not any("lowered_module" in n.name for n in ep.graph.nodes)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 4, 8, 8), (2, 3), id="Dim 2, 3 (supported), channels = 5 (unsupported)."
        ),
    ],
)
def test_mean_dim__channels_first__unsupported_channels(
    input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimConvModule(
        dim, keepdim, out_channels=5
    )  # Only multiples of 8 (num_macs) are supported.

    # Run conversion
    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [MeanDim])


class MaxPoolMeanDimModule(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim, self.keepdim = dim, keepdim

    def forward(self, x):
        x = torch.max_pool2d(
            x, kernel_size=1
        )  # NoOp, but it enforces the channels first format.
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class TestMeanDimNewNeutronFlow:

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
            expected_delegated_ops = {MeanDim: 1}

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

        # Make sure the `mean` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [MeanDim])

    @pytest.fixture(params=[True, False], ids=lambda keep_dim: f"keep_dim = {keep_dim}")
    def keep_dim(self, request):
        return request.param

    def test__basic_nsys_inference__qat(self, mocker, use_qat, keep_dim):
        input_shape = (23,)
        model = MeanDimModule(0, keep_dim)
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((5,), 0, id="1D, dim = 0."),
            pytest.param((4, 2), 0, id="2D, dim = 0."),
            pytest.param((4, 2), -1, id="2D, dim = -1."),
            pytest.param((3, 1, 4), 2, id="3D, dim = 2."),
            pytest.param((1, 3, 3, 7), 3, id="4D, dim = 3."),
            pytest.param((3, 1, 4, 1, 5), -1, id="5D, dim = -1."),
            pytest.param((3, 1, 4, 1, 5), 0, id="5D, dim = 0."),
        ],
    )
    def test__single_dims(self, mocker, input_shape, dim, keep_dim):
        model = MeanDimModule(dim, keep_dim)
        # Relatively large error, but it is actually equal to the output scale, so it is a single bit error.
        # TODO Replace with quantized dataset testing and `atol = 1`.
        atol = 0.014
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((4, 2), (-2,), id="2D, dim = (-2,)."),
            pytest.param((2, 3, 4), (0, 2), id="3D, dim = (0, 2,)."),
            pytest.param((1, 3, 3, 7), (2, -3), id="4D, dim = (2, -3)."),
            pytest.param((3, 1, 4, 1, 5), (3, -5, -4), id="5D, dim = (3, -5 ,-4)."),
        ],
    )
    def test__tuple_dims(self, mocker, input_shape, dim, keep_dim):
        model = MeanDimModule(dim, keep_dim)
        # Relatively large error, but it is actually equal to the output scale, so it is a single bit error.
        # TODO Replace with quantized dataset testing and `atol = 1`.
        atol = 0.015
        self.assert_delegated(model, input_shape, mocker, atol=atol)

    def test__compute_error(self, mocker, keep_dim):
        input_shape, dim = (1, 3, 3, 7), -2
        model = MeanDimModule(dim, keep_dim)

        # Neutron produces an incorrect result in this case (maximum absolute error ~= 0.0607 (more than 2 * scale)).
        # This test detects the failure to alert us once the bug is fixed. It should be fixed in Neutron 3.1.2.
        with pytest.raises(AssertionError):
            self.assert_delegated(model, input_shape, mocker, atol=0.06)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((3, 1, 4), 1, id="3D, dim = 1."),
            pytest.param((3, 1, 4, 1, 5), -2, id="5D, dim = -2."),
        ],
    )
    def test__noop__only_node__not_delegated(self, input_shape, dim):
        keep_dim = True  # Reduction over a dimension of size `1` with `keep_dim=True` is a no-op.
        model = MeanDimModule(dim, keep_dim)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((3, 1, 4), 1, id="3D, dim = 1."),
            pytest.param((3, 1, 4, 1, 5), -2, id="5D, dim = -2."),
        ],
    )
    def test__noop__not_only_node__delegated(self, mocker, input_shape, dim):
        keep_dim = True  # Reduction over a dimension of size `1` with `keep_dim=True` is a no-op.
        model = MeanDimAddModule(dim, keep_dim)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={MeanDim: 1, AddTensor: 1},
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((3, 1, 4), 1, id="3D, dim = 1."),
            pytest.param((3, 1, 4, 1, 5), -2, id="5D, dim = -2."),
        ],
    )
    def test__no_reduction__keepdim_false__delegated(self, mocker, input_shape, dim):
        # These cases reduce over a dimension of size 1.
        # When `keep_dim=True` the node is a noop, and it's not delegated (see `test__noop__only_node__not_delegated`),
        # but with `keep_dim=False` it changes the shape so it's not a noop and is therefore delegated successfully.
        keep_dim = False
        model = MeanDimModule(dim, keep_dim)
        self.assert_delegated(model, input_shape, mocker)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [((1, 7, 3, 3), 1)],
        ids=lambda val: f"shape={val}" if isinstance(val, tuple) else f"dim={val}",
    )
    def test__channels_first(self, mocker, input_shape, dim, keep_dim):
        # Just 1 test case to verify correct handling of the `dim`.
        # Most cases fall into the single bit error case, and since this test uses 2 operators, the error accumulates
        #  and the final error is larger. We cannot with 100% certainty say that the error is only caused by the single
        #  bit errors and not related to the format. That's why only this 1 case with no errors is used.
        model = MaxPoolMeanDimModule(dim, keep_dim)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1, MeanDim: 1},
        )
