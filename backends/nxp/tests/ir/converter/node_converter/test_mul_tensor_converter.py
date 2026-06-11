# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import MulTensorConvModule, MulTensorModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    Convolution,
    ExecutorchDelegateCall,
    MulTensor,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestMulTensor:
    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 8), id="2D."),
            pytest.param((1, 4, 8), id="3D."),
            pytest.param((1, 4, 8, 8), id="4D."),
        ],
    )
    def test__basic_nsys_inference(self, mocker, request, x_input_shape):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = MulTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={MulTensor: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            request,
        )

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1, 4, 8), id="3D."),
            pytest.param((1, 4, 8, 8), id="4D."),
        ],
    )
    def test__basic_nsys_inference_qat(self, mocker, request, x_input_shape):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = MulTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={MulTensor: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            request,
            use_qat=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 6)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((5, 3, 4)), ModelInputSpec((1, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((4,)), ModelInputSpec((4, 4))], id="2 inputs 1D+2D."
            ),
            pytest.param(
                [ModelInputSpec((10,)), ModelInputSpec((1, 1))],
                id="2 inputs 2D, num_elems of input == num_elems of output",
            ),
        ],
    )
    def test__correct_broadcast(self, input_spec, mocker, request):
        model = MulTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={MulTensor: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_spec, graph_verifier, request)

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
    def test__incorrect_broadcast(self, input_spec):
        # Broadcast where at least one of the inputs is not equal to output is not supported
        model = MulTensorModule()

        delegated_ep = to_quantized_edge_program(model, input_spec).exported_program()

        # Make sure the `mul.Tensor` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [MulTensor])

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param(
                (1, 4, 5, 5), id="4D, product of dims is not a multiple of 8."
            ),
        ],
    )
    def test__w_conv(self, mocker, request, x_input_shape):
        model = MulTensorConvModule()

        n, c, h, w = x_input_shape
        y_input_spec = ModelInputSpec((n, 8, h, w))
        x_input_spec = ModelInputSpec(x_input_shape)

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MulTensor: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            [x_input_spec, y_input_spec],
            graph_verifier,
            request,
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
        model = MulTensorConvModule()

        delegated_ep = to_quantized_edge_program(model, input_spec).exported_program()

        # Make sure the `mul.Tensor` was NOT delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
        assert graph_contains_any_of_ops(delegated_ep.graph, [MulTensor])
