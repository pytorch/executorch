# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import SubTensorConvModule, SubTensorModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    Convolution,
    ExecutorchDelegateCall,
    SubTensor,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestSubTensor:
    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 5), id="2D."),
            pytest.param((1, 4, 7), id="3D."),
            pytest.param(
                (6, 82),
                id="2D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (1, 68, 7),
                id="3D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (2, 4, 3, 15),
                id="4D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (1, 4, 9, 11, 4),
                id="5D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__basic_nsys_inference(self, x_input_shape, mocker):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = SubTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={SubTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            dataset_creator,
        )

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 5), id="2D."),
            pytest.param((2, 4, 3, 15), id="4D."),
            pytest.param(
                (1, 4, 7),
                id="3D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (1, 4, 9, 11, 4),
                id="5D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__basic_nsys_inference_qat(self, x_input_shape, mocker):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = SubTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={SubTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            dataset_creator,
            use_qat=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 6)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((4,)), ModelInputSpec((4, 4))], id="2 inputs 1D + 2D."
            ),
            pytest.param(
                [ModelInputSpec((5, 3, 4)), ModelInputSpec((1, 3, 1))],
                id="2 inputs 3D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                [ModelInputSpec((69, 73)), ModelInputSpec((1, 73))],
                id="2 inputs 2D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__broadcast(self, input_spec, mocker):
        model = SubTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={SubTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            dataset_creator,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 1)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((1, 3, 4)), ModelInputSpec((5, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((6, 4)), ModelInputSpec((6, 6, 1))],
                id="2 inputs 2D+3D.",
            ),
        ],
    )
    def test__broadcast_unsupported(self, input_spec):
        # Broadcast where at least one of the inputs is not equal to output is not supported
        model = SubTensorModule()

        delegated_ep = to_quantized_edge_program(model, input_spec).exported_program()

        # Make sure the `sub.Tensor` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [SubTensor])

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param(
                (1, 4, 5, 5), id="4D, product of dims is not a multiple of 8."
            ),
        ],
    )
    def test__w_conv(self, x_input_shape, mocker):
        model = SubTensorConvModule()

        n, c, h, w = x_input_shape
        y_input_spec = ModelInputSpec((n, 8, h, w))
        x_input_spec = ModelInputSpec(x_input_shape)

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={SubTensor: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, y_input_spec],
            graph_verifier,
            dataset_creator,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((1, 4, 7, 1)), ModelInputSpec((1, 8, 1, 1))],
                id="2 inputs 4D + 4D.",
            ),
            pytest.param(
                [ModelInputSpec((1, 4, 5, 5)), ModelInputSpec((1, 8, 5, 1))],
                id="2 inputs 4D + 4D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__w_conv_broadcast(self, input_spec, mocker):
        model = SubTensorConvModule()
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={SubTensor: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            dataset_creator,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((1, 4, 5, 5)), ModelInputSpec((1, 5))],
                id="2 inputs 4D + 2D.",
            ),
            pytest.param(
                [ModelInputSpec((1, 4, 4, 10)), ModelInputSpec((1, 4, 1))],
                id="2 inputs 4D + 3D.",
            ),
        ],
    )
    def test__w_conv_unsupported(self, input_spec):
        model = SubTensorConvModule()

        delegated_ep = to_quantized_edge_program(model, input_spec).exported_program()

        # Make sure the `sub.Tensor` was NOT delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
        assert graph_contains_any_of_ops(delegated_ep.graph, [SubTensor])
