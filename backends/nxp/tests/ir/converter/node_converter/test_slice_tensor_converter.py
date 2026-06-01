# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
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

from executorch.backends.nxp.tests.models import (
    SliceTensorConvModule,
    SliceTensorModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    Convolution,
    ExecutorchDelegateCall,
    Slice,
    SliceCopy,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


passing_cases = [
    pytest.param((24, 32), (0, 1), (0, 16), (24, 32), id="2D, no transpose"),
    pytest.param(
        (24, 32, 64), (0, 1, 2), (0, 0, 8), (24, 32, 64), id="3D, no transpose"
    ),
    pytest.param(
        (24, 32, 64, 48),
        (0, 1, 2, 3),
        (0, 0, 0, 8),
        (24, 32, 64, 48),
        id="4D, no transpose",
    ),
    pytest.param(
        (24, 32),
        (0, 1),
        (0, 13),
        (24, 32),
        id="2D, start arg not divisible by num_macs",
    ),
    pytest.param(
        (24, 32),
        (0, 1),
        (0, 0),
        (24, 31),
        id="2D, end arg not divisible by num_macs",
    ),
    pytest.param((24, 32), (1, 0), (16, 0), (32, 24), id="2D, mixed dim args"),
    pytest.param((24, 32), (0, -1), (0, 16), (24, 32), id="2D, negative dim arg"),
]

xfail_cases = [
    pytest.param(
        (24, 32),
        (0, 1),
        (8, 0),
        (24, 32),
        id="2D, one transpose",
        marks=pytest.mark.xfail(
            reason="Neutron-converter now only supports transpose in 4D, ticket: AIR-13446",
            strict=True,
        ),
    ),
    pytest.param(
        (24, 32, 64),
        (0, 1, 2),
        (0, 8, 0),
        (24, 32, 64),
        id="3D, one transpose",
        marks=pytest.mark.xfail(
            reason="Neutron-converter now only supports transpose in 4D, ticket: AIR-13446",
            strict=True,
        ),
    ),
    pytest.param(
        (24, 32, 64, 48),
        (0, 1, 2, 3),
        (0, 0, 8, 0),
        (24, 32, 64, 48),
        id="4D, one transpose",
        marks=pytest.mark.xfail(
            reason="Neutron-converter now only supports transpose of NHWC -> NCHW and vice versa, ticket: AIR-13446",
            strict=True,
        ),
    ),
    pytest.param(
        (24, 32, 64),
        (0, 1, 2),
        (8, 8, 0),
        (24, 32, 64),
        id="3D, two transposes",
        marks=pytest.mark.xfail(
            reason="Neutron-converter now only supports transpose in 4D, ticket: AIR-13446",
            strict=True,
        ),
    ),
    pytest.param(
        (24, 32, 64, 48),
        (0, 1, 2, 3),
        (16, 0, 8, 0),
        (24, 32, 64, 48),
        id="4D, two transposes",
        marks=pytest.mark.xfail(
            reason="Bug in neutron-converter, ticket: AIR-13665", strict=True
        ),
    ),
    pytest.param(
        (24, 32, 64, 48),
        (0, 1, 2, 3),
        (16, 0, 8, 0),
        (24, 24, 56, 48),
        id="4D, three transposes",
        marks=pytest.mark.xfail(
            reason="Bug in neutron-converter, ticket: AIR-13665", strict=True
        ),
    ),
]


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    passing_cases + xfail_cases,
)
def test_slice_tensor_quant_conversion(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, x_input_shape).exported_program()

    # Check if slices were delegated
    assert not graph_contains_any_of_ops(edge_program.graph, [Slice, SliceCopy])
    assert graph_contains_any_of_ops(edge_program.graph, [ExecutorchDelegateCall])

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data = {0: input_data}

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tfl_model=tflite_flatbuffers_model,
    )


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (1, 16, 32, 48),
            (0, 1, 2, 3),
            (0, 8, 0, 0),
            (1, 16, 32, 48),
            id="4D, handle channel order swap",
        )
    ],
)
def test_slice_tensor_w_conv_quant_conversion(
    mocker, x_input_shape, dims, starts, ends
):
    in_channels = out_channels = x_input_shape[1]
    model = SliceTensorConvModule(
        dims=dims,
        starts=starts,
        ends=ends,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, x_input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Check if slices were delegated
    assert not graph_contains_any_of_ops(edge_program.graph, [Slice, SliceCopy])
    assert graph_contains_any_of_ops(edge_program.graph, [ExecutorchDelegateCall])

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data = {0: input_data}

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (24, 32), (0, 1), (0, 16), (24, 8), id="2D, start is higher than end"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 16), (24, 16), id="2D, start is equal to end"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 32), (24, 32), id="2D, start is equal to size"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 0), (24, -35), id="2D, clipped end equal to zero"
        ),
        pytest.param(
            (24, 32), (0, 1), (64, 0), (24, 32), id="2D, clipped start equal to size"
        ),
    ],
)
def test_invalid_slice(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, x_input_shape).exported_program()

    # Capture generated model, should be None because the model is invalid
    assert converter_spy.spy_return is None


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (24, 31),
            (0, 1),
            (0, 0),
            (24, 16),
            id="2D, input shape not divisible by num_macs",
        ),
        pytest.param(
            (24, 26, 64),
            (0, 1, 2),
            (0, 4, 0),
            (24, 26, 64),
            id="3D, input shape not divisible by num_macs",
        ),
    ],
)
def test_slice_not_delegated(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    edge_program = to_quantized_edge_program(model, x_input_shape).exported_program()
    nodes = list(edge_program.graph.nodes)

    num_slice_ops = 0
    for i in range(len(x_input_shape)):
        if starts[i] != 0 or ends[i] != x_input_shape[i]:
            num_slice_ops += 1

    for i in range(0, num_slice_ops):
        slice_idx = (i + 1) * 3
        assert nodes[slice_idx].target in [Slice, SliceCopy]


class TestSliceTensorConverterNewNeutronFlow:
    @staticmethod
    def _slice_id(prefix, input_shape, dims, starts, ends):
        return f"{prefix}rank={len(input_shape)}_dims={str(dims)}_starts={str(starts)}_ends={str(ends)}"

    @staticmethod
    def assert_delegated_and_correct(model, input_shape, num_slices, mocker, use_qat):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={SliceCopy: num_slices},
            expected_non_delegated_ops={},
        )
        dataset = RandomDatasetCreator(low=-255.0, high=255.0)
        comparator = AllCloseOutputComparator()

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset,
            comparator,
            use_new_flow_neutron_c=True,
            use_qat=use_qat,
        )

    @staticmethod
    def assert_model_without_slices(model, input_shape):
        delegated_ep = to_quantized_edge_program(
            model, input_shape, use_new_flow_neutron_c=True
        ).exported_program()

        # Check there are no slices and nothing is delegated
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert not graph_contains_any_of_ops(delegated_ep.graph, [Slice, SliceCopy])

    @staticmethod
    def assert_not_delegated(model, input_shape):
        delegated_ep = to_quantized_edge_program(
            model, input_shape, use_new_flow_neutron_c=True
        ).exported_program()

        # Make sure the `slice` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [Slice, SliceCopy])

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (5, 2, 3, 4),
                d := (0,),
                s := (1,),
                e := (4,),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5, 5, 3, 4),
                d := (0, 1),
                s := (1, 1),
                e := (4, 3),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (7, 13, 5, 15),
                d := (0, 1, 2, 3),
                s := (4, 3, 1, 8),
                e := (5, 10, 4, 11),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5, 13, 5, 13),
                d := (0, 1, 2, 3),
                s := (0, 0, 0, 0),
                e := (4, 11, 4, 11),
                id=_slice_id("basic, right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (7, 13, 3, 15),
                d := (0, 1, 2, 3),
                s := (2, 5, 1, 4),
                e := ins,
                id=_slice_id("basic, left trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (7, 4, 7),
                d := (0, 1, 2),
                s := (1, 1, 3),
                e := (6, 3, 5),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (4, 5, 9),
                d := (0, 1, 2),
                s := (0, 0, 0),
                e := (3, 4, 7),
                id=_slice_id("basic, right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (4, 7, 9),
                d := (0, 1, 2),
                s := (3, 2, 2),
                e := ins,
                id=_slice_id("basic, left trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (4, 5),
                d := (0, 1),
                s := (1, 1),
                e := (2, 4),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (4, 5),
                d := (0, 1),
                s := (0, 0),
                e := (2, 4),
                id=_slice_id("basic, right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (4, 5),
                d := (0, 1),
                s := (1, 2),
                e := ins,
                id=_slice_id("basic, left trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5,),
                d := (0,),
                s := (1,),
                e := (4,),
                id=_slice_id("basic, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5,),
                d := (0,),
                s := (0,),
                e := (4,),
                id=_slice_id("basic, right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5,),
                d := (0,),
                s := (1,),
                e := ins,
                id=_slice_id("basic, left trimmed:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__basic(self, input_shape, dims, starts, ends, mocker):
        model = SliceTensorModule(dims, starts, ends)

        num_slices = len(dims)
        self.assert_delegated_and_correct(
            model, input_shape, num_slices, mocker, use_qat=False
        )

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (4, 2, 7, 4),
                d := (2,),
                s := (5,),
                e := (6,),
                id=_slice_id("edge case, dimension reduced to 1:", ins, d, s, e),
            ),
            pytest.param(
                ins := (11, 2, 7, 5),
                d := (2,),
                s := (6,),
                e := (6,),
                id=_slice_id("edge case, dimension reduced to 0:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__reduction(self, input_shape, dims, starts, ends, mocker):
        model = SliceTensorModule(dims, starts, ends)

        slice_lengths = [e - s for s, e in zip(starts, ends)]
        if all(sl == 0 for sl in slice_lengths):
            # reductions to 0 are disabled in the backend
            self.assert_not_delegated(model, input_shape)
        else:
            num_slices = len(dims)
            self.assert_delegated_and_correct(
                model, input_shape, num_slices, mocker, use_qat=False
            )

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (5, 2, 3, 4),
                d := (0,),
                s := (-12,),
                e := (2,),
                id=_slice_id("edge case, `start` clipped:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5, 7, 5, 7),
                d := (0,),
                s := (1,),
                e := (12,),
                id=_slice_id("edge case, `end` clipped:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__clipped(self, input_shape, dims, starts, ends, mocker):
        model = SliceTensorModule(dims, starts, ends)

        num_slices = len(dims)
        self.assert_delegated_and_correct(
            model, input_shape, num_slices, mocker, use_qat=False
        )

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (5, 11, 13, 3),
                d := (1,),
                s := (-5,),
                e := (10,),
                id=_slice_id("edge case, `start` normalized:", ins, d, s, e),
            ),
            pytest.param(
                ins := (7, 15, 5, 7),
                d := (1,),
                s := (2,),
                e := (-2,),
                id=_slice_id("edge case, `end` normalized:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__normalization(
        self, input_shape, dims, starts, ends, mocker
    ):
        model = SliceTensorModule(dims, starts, ends)

        num_slices = len(dims)
        self.assert_delegated_and_correct(
            model, input_shape, num_slices, mocker, use_qat=False
        )

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (5000, 3, 5, 3),
                d := (0,),
                s := (1250,),
                e := (2500,),
                id=_slice_id("big args, left and right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (2, 5000, 5, 3),
                d := (1,),
                s := (0,),
                e := (4999,),
                id=_slice_id("big args, right trimmed:", ins, d, s, e),
            ),
            pytest.param(
                ins := (2, 3, 5000, 3),
                d := (2,),
                s := (1,),
                e := (5000,),
                id=_slice_id("big args, left trimmed:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__big(self, input_shape, dims, starts, ends, mocker):
        model = SliceTensorModule(dims, starts, ends)

        num_slices = len(dims)
        self.assert_delegated_and_correct(
            model, input_shape, num_slices, mocker, use_qat=False
        )

    @pytest.mark.parametrize(
        "input_shape, dims, starts, ends",
        [
            pytest.param(
                ins := (5, 2, 3, 4),
                d := (2,),
                s := (0,),
                e := (3,),
                id=_slice_id("edge case, one dimension identity:", ins, d, s, e),
            ),
            pytest.param(
                ins := (5, 2, 3, 4),
                d := (0, 1, 2, 3),
                s := (0, 0, 0, 0),
                e := ins,
                id=_slice_id("edge case, all dimensions identity:", ins, d, s, e),
            ),
        ],
    )
    def test_nsys_inference__identity(self, input_shape, dims, starts, ends):
        model = SliceTensorModule(dims, starts, ends)

        self.assert_model_without_slices(model, input_shape)

    def test_nsys_inference__with_conv(self, mocker):
        input_shape = (11, 13, 5, 7)
        in_channels = input_shape[1]
        out_channels = 19

        # we test functionality on `channels` dim
        dims = (1,)
        starts = (2,)
        ends = (out_channels - 2,)
        model = SliceTensorConvModule(dims, starts, ends, in_channels, out_channels)

        num_slices = len(dims)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={SliceCopy: num_slices},
            expected_non_delegated_ops={Convolution: 1},
        )
        dataset = RandomDatasetCreator(low=-255.0, high=255.0)
        comparator = AllCloseOutputComparator()

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset,
            comparator,
            use_new_flow_neutron_c=True,
            use_qat=False,
        )

    def test_nsys_inference__qat(self, mocker):
        input_shape = (7, 13, 7, 9)
        dims = (0, 1, 2, 3)
        starts = (1, 2, 3, 2)
        ends = (6, 10, 5, 8)

        model = SliceTensorModule(dims, starts, ends)

        num_slices = len(dims)
        self.assert_delegated_and_correct(
            model, input_shape, num_slices, mocker, use_qat=True
        )
