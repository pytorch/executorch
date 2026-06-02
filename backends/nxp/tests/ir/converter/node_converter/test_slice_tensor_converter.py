# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
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


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


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
