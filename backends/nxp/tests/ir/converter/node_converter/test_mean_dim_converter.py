# Copyright 2025-2026 NXP
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
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.max_pool_2d_options import (
    MaxPool2D,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.mean_options import (
    Mean,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.transpose_options import (
    Transpose,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
    MeanDim,
)
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


class MaxPoolMeanDimModule(torch.nn.Module):
    @staticmethod
    def noop_max_pool_2d(x):
        """Call `torch.max_pool2d` that is a NoOp, but it enforces the ChannelsFirst format in the `NodeFormatInference`."""
        return torch.max_pool2d(x, kernel_size=1)

    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim, self.keepdim = dim, keepdim

    def forward(self, x):
        x = self.noop_max_pool_2d(x)
        x = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        return x


class MeanDimMaxPoolModule(MaxPoolMeanDimModule):
    def forward(self, x):
        x = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        x = self.noop_max_pool_2d(x)
        return x


def assert_delegated(
    model,
    input_shape,
    mocker,
    use_qat=False,
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

    remove_quant_io_ops = True  # Use quantized dataset.
    output_comparator = AllCloseOutputComparator(atol=1)  # Allow single bit error.

    lower_run_compare(
        model,
        input_shape,
        graph_verifier,
        dataset_creator,
        output_comparator,
        use_qat=use_qat,
        remove_quant_io_ops=remove_quant_io_ops,
    )


def assert_not_delegated(model, input_shape):
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `mean` was NOT delegated.
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [MeanDim])


class TestMeanDim:

    @pytest.fixture(params=[True, False], ids=lambda keep_dim: f"keep_dim = {keep_dim}")
    def keep_dim(self, request):
        return request.param

    def test__basic_nsys_inference__qat(self, mocker, use_qat, keep_dim):
        input_shape = (23,)
        model = MeanDimModule(0, keep_dim)
        assert_delegated(model, input_shape, mocker, use_qat=use_qat)

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
        assert_delegated(model, input_shape, mocker)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((4, 2), (-2,), id="2D, dim = (-2,)."),
            pytest.param((2, 3, 4), (0, 2), id="3D, dim = (0, 2,)."),
            pytest.param((1, 3, 3, 7), (2, -3), id="4D, dim = (2, -3)."),
            pytest.param((1, 3, 3, 7), -2, id="4D, dim = -2."),
            pytest.param((3, 1, 4, 1, 5), (3, -5, -4), id="5D, dim = (3, -5 ,-4)."),
        ],
    )
    def test__tuple_dims(self, mocker, input_shape, dim, keep_dim):
        model = MeanDimModule(dim, keep_dim)
        assert_delegated(model, input_shape, mocker)

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
        assert_not_delegated(model, input_shape)

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
        assert_delegated(
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
            pytest.param((1, 7, 3, 3), [0], id="4D, dim = [0]."),
        ],
    )
    def test__no_reduction__keepdim_false__delegated(self, mocker, input_shape, dim):
        # These cases reduce over a dimension of size 1.
        # When `keep_dim=True` the node is a noop, and it's not delegated (see `test__noop__only_node__not_delegated`),
        # but with `keep_dim=False` it changes the shape so it's not a noop and is therefore delegated successfully.
        keep_dim = False
        model = MeanDimModule(dim, keep_dim)
        assert_delegated(model, input_shape, mocker)

    def test__channels_first__keep_dim__true(self, mocker):
        # Just 1 test case to verify correct handling of the `dim`.
        # Most cases fall into the single bit error case, and since this test uses 2 operators, the error accumulates
        #  and the final error is larger. We cannot with 100% certainty say that the error is only caused by the single
        #  bit errors and not related to the format. That's why only this 1 case with no errors is used.
        input_shape, dim = (1, 7, 3, 3), 1
        model = MaxPoolMeanDimModule(dim, True)
        assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1, MeanDim: 1},
        )

    class TestKeepDimFalseFormatHandling:
        """When `keep_dim = False`, the `mean.dim` operator changes the rank, so the format have to be explicitly
        handled. The tests in this class focus on the related edge cases.
        """

        def _assert_neutron_ir_model_has_ops(
            self, model_builder_finish_spy, expected_ops
        ):
            assert (
                model_builder_finish_spy.call_count == 1
            ), "Conversion to Neutron IR happened multiple times."

            neutron_ir_ops = model_builder_finish_spy.spy_return.sub_graphs[
                0
            ].operators.vector
            assert len(neutron_ir_ops) == len(
                expected_ops
            ), "Neutron IR model doesn't have the expected number of ops."

            for op, expected_op in zip(neutron_ir_ops, expected_ops, strict=True):
                assert isinstance(
                    op.builtin_options, expected_op
                ), f"Expected {expected_op}, got {op}."

        @pytest.mark.parametrize(
            "dim",
            [
                1,
                [0, -3],
                (-4, 1, 2),
                [-3, 3],
                [1, 2, 3],
            ],
            ids=lambda dim: f"dim={dim}",
        )
        def test__channels_first_input__reducing_channels(self, mocker, dim):
            # If the channels dimension is reduced (removed), the `mean` output will always be equal in channels first
            #  and channels last, so no `Transpose` ops are added.
            input_shape = (1, 7, 3, 3)
            model = MaxPoolMeanDimModule(dim, False)

            model_builder_finish_spy = mocker.spy(ModelBuilder, "finish")
            assert_delegated(
                model,
                input_shape,
                mocker,
                expected_delegated_ops={
                    MaxPool2DWithIndices: 1,
                    GetItem: 1,
                    MeanDim: 1,
                },
            )
            self._assert_neutron_ir_model_has_ops(
                model_builder_finish_spy,
                expected_ops=[
                    Transpose,
                    MaxPool2D,
                    Mean,
                ],
            )

        @pytest.mark.parametrize(
            "dim",
            [
                (2, 3),
                [1, -2, 3],
                [-1, -2, 0],
            ],
            ids=lambda dim: f"dim={dim}",
        )
        def test__channels_first_input__reducing_all_spatial_dims(self, mocker, dim):
            # If tall he spatial dimensions are reduced (removed), the `mean` output will always be equal in channels
            #  first and channels last, so no `Transpose` ops are added.
            input_shape = (1, 7, 3, 3)
            model = MaxPoolMeanDimModule(dim, False)

            model_builder_finish_spy = mocker.spy(ModelBuilder, "finish")
            assert_delegated(
                model,
                input_shape,
                mocker,
                expected_delegated_ops={
                    MaxPool2DWithIndices: 1,
                    GetItem: 1,
                    MeanDim: 1,
                },
            )
            self._assert_neutron_ir_model_has_ops(
                model_builder_finish_spy,
                expected_ops=[
                    Transpose,
                    MaxPool2D,
                    Mean,
                ],
            )

        @pytest.mark.xfail(strict=True, reason="Known Neutron bug (AIR-14726).")
        @pytest.mark.parametrize(
            "dim",
            [
                0,
                (2,),
                [-1, 0],
            ],
            ids=lambda dim: f"dim={dim}",
        )
        def test__channels_first_input__not_reducing_channels_or_all_spatial_dims(
            self, mocker, dim
        ):
            # If the channels dimension is not reduced, a `Transpose` operator must be added to make the input channels
            #  first in Neutron IR.

            input_shape = (1, 7, 3, 3)
            model = MaxPoolMeanDimModule(dim, False)

            model_builder_finish_spy = mocker.spy(ModelBuilder, "finish")
            assert_delegated(
                model,
                input_shape,
                mocker,
                expected_delegated_ops={
                    MaxPool2DWithIndices: 1,
                    GetItem: 1,
                    MeanDim: 1,
                },
            )

            self._assert_neutron_ir_model_has_ops(
                model_builder_finish_spy,
                expected_ops=[
                    Transpose,
                    MaxPool2D,
                    Transpose,  # The necessary `Transpose` operator.
                    Mean,
                ],
            )

        @pytest.mark.parametrize(
            "input_shape, dim",
            [
                pytest.param((2, 3, 4, 5, 6), 0, id="dim=0, 5D->4D"),
                pytest.param((2, 3, 4, 5, 6), [-3], id="dim=[-3], 5D->4D"),
                pytest.param((1, 2, 3, 4, 5, 6), (1, -1), id="dim=(1, -1), 6D->4D"),
            ],
            ids=lambda dim: f"dim={dim}",
        )
        def test__channels_first_output(self, mocker, input_shape, dim):
            model = MeanDimMaxPoolModule(dim, False)

            model_builder_finish_spy = mocker.spy(ModelBuilder, "finish")
            assert_delegated(
                model,
                input_shape,
                mocker,
                expected_delegated_ops={
                    MaxPool2DWithIndices: 1,
                    GetItem: 1,
                    MeanDim: 1,
                },
            )

            self._assert_neutron_ir_model_has_ops(
                model_builder_finish_spy,
                expected_ops=[
                    Mean,
                    Transpose,  # The necessary `Transpose` operator.
                    MaxPool2D,
                    Transpose,
                ],
            )
