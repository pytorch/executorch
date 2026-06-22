# Copyright 2024,2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.backend.edge_program_converter import exir_ops
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dModule, LinearModule, ReLUModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddMm,
    Convolution,
    DequantizePerChannel,
    DequantizePerTensor,
    PermuteCopy,
    QuantizePerTensor,
    Relu,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
ReLU = exir_ops.edge.aten.relu.default


class ConvReLUModule(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()

        self.conv = Conv2dModule(in_channels=in_channels, out_channels=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class LinearReLUModule(torch.nn.Module):
    def __init__(self, in_features: int = 32, out_features: int = 16):
        super().__init__()

        self.linear = LinearModule(
            bias=True, in_features=in_features, out_features=out_features
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


class TestReLU:
    @pytest.mark.parametrize(
        ["model", "input_shape"],
        [
            pytest.param(
                lambda: LinearReLUModule(in_features=9, out_features=17),
                (9, 9),
                id="Linear(1D-in): num_channels not divisible by NUM_MACS",
            ),
            pytest.param(
                lambda: LinearReLUModule(in_features=9, out_features=15),
                (1, 7, 9),
                id="Linear(2D-in): num_channels not divisible by NUM_MACS",
            ),
            pytest.param(
                lambda: LinearReLUModule(in_features=8, out_features=16),
                (1, 8, 8),
                id="Linear(2D-in): num_channels divisible by NUM_MACS",
            ),
            pytest.param(
                lambda: LinearReLUModule(in_features=9, out_features=15),
                (1, 9, 9, 9),
                id="Linear(3D-in): num_channels not divisible by NUM_MACS",
            ),
            pytest.param(
                lambda: ConvReLUModule(in_channels=17, out_channels=9),
                (1, 17, 9, 9),
                id="Conv: num_channels not divisible by NUM_MACS",
            ),
            pytest.param(
                lambda: ConvReLUModule(in_channels=8, out_channels=16),
                (1, 8, 8, 8),
                id="Conv: num_channels divisible by NUM_MACS",
            ),
        ],
    )
    def test_relu_conversion__full_pipeline(self, mocker, request, model, input_shape):
        model = model()  # Avoid model creation at import time
        is_conv_module = not hasattr(model, "linear")

        graph_verifier = DetailedGraphVerifier(
            mocker=mocker,
            expected_delegated_ops=(
                {Convolution: 1, Relu: 1} if is_conv_module else {AddMm: 1, Relu: 1}
            ),
            expected_non_delegated_ops={},
            ops_to_ignore={
                PermuteCopy,
                ViewCopy,
                QuantizePerTensor,
                DequantizePerTensor,
                DequantizePerChannel,
            },
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param(
                (3, 9, 9),
                id="num_channels not divisible by NUM_MACS, alone in partition",
            ),
            pytest.param(
                (1, 17, 17),
                id="num_channels not divisible by NUM_MACS, alone in partition",
            ),
        ],
    )
    def test_relu_conversion__non_delegated_with_old_flow(
        self, mocker, request, input_shape
    ):
        verifier = DetailedGraphVerifier(
            mocker=mocker,
            expected_delegated_ops={Relu: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            ReLUModule(),
            input_shape,
            verifier,
            request,
            RandomDatasetCreator(low=-1, high=1),
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param(
                (3, 9, 9),
                id="num_channels not divisible by NUM_MACS, alone in partition",
            ),
            pytest.param(
                (1, 17, 17),
                id="num_channels not divisible by NUM_MACS, alone in partition",
            ),
        ],
    )
    def test_relu_conversion__no_delegated_node_when_noop(self, input_shape):
        def generate_calibration_data(input_spec):
            return [
                # Generate inputs in range <0, 1> - ReLU degrades to identity
                tuple([torch.rand(spec.shape, dtype=spec.dtype) for spec in input_spec])
                for _ in range(4)
            ]

        # Run conversion
        delegated_ep = to_quantized_edge_program(
            ReLUModule(),
            input_shape,
            delegate_to_npu=True,
            get_calibration_inputs_fn=generate_calibration_data,
        ).exported_program()

        # Ensure identity ReLU was not delegated
        assert graph_contains_any_of_ops(delegated_ep.graph, [ReLU])
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
