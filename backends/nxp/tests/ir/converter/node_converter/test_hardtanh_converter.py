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
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import (
    AtenModelBuilderDirector,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator as Ops,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dWithActivation, HardTanhModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    Convolution,
    ExecutorchDelegateCall,
    HardTanh,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class AddHardTanhModule(HardTanhModule):
    def forward(self, x):
        x = x + x
        x = super().forward(x)
        return x


class TestHardTanh:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self, model, input_shape, mocker, use_qat=False, expected_delegated_ops=None
    ):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=(
                expected_delegated_ops
                if expected_delegated_ops is not None
                else {HardTanh: 1}
            ),
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "activation_range",
        [
            (-1, 3),
            (0, float("inf")),
        ],
    )
    @pytest.mark.parametrize(
        "inplace", [True, False], ids=lambda ip: "Inplace" if ip else "Not inplace"
    )
    def test__qat(
        self, mocker, activation_range: tuple[float, float], use_qat, inplace
    ):
        input_shape = (23,)
        model = HardTanhModule(*activation_range, inplace)

        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "inplace", [True, False], ids=lambda ip: "Inplace" if ip else "Not inplace"
    )
    def test__from_relu6__after_conv(self, mocker, inplace: bool):
        # The torch.nn.Relu6 inherits from torch.nn.Hardtanh, and hence represented as HardTanh in ATen.
        # Testing the hardtanh originated from torch.nn.Relu6 op.
        input_shape = (1, 3, 4, 5)
        model = Conv2dWithActivation(
            activation=torch.nn.ReLU6(inplace=inplace),
            in_channels=input_shape[1],
            out_channels=2,
        )

        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={HardTanh: 1, Convolution: 1},
        )

    @pytest.mark.parametrize(
        "activation_range",
        [
            (0.0, 6.0),
            (-1.0, 1),
            (0, 1),
            (0.0, float("inf")),
        ],
    )
    @pytest.mark.parametrize(
        "inplace", [True, False], ids=lambda ip: "Inplace" if ip else "Not inplace"
    )
    def test__hardtanh__mappable_to_relu__after_conv(
        self,
        mocker,
        activation_range: tuple[float, float],
        inplace: bool,
    ):
        input_shape = (1, 3, 4, 5)
        model = Conv2dWithActivation(
            activation=torch.nn.Hardtanh(*activation_range, inplace),
            in_channels=input_shape[1],
            out_channels=2,
        )

        self.assert_delegated(
            model,
            input_shape,
            mocker,
            expected_delegated_ops={HardTanh: 1, Convolution: 1},
        )

    @pytest.mark.parametrize(
        "activation_range",
        [
            (-1, 3),
            (2.27, 3.14),
            (-0.1, 0),
            (float("-inf"), 1.23),
        ],
    )
    def test__hardtanh__not_mappable_to_relu(
        self,
        mocker,
        activation_range: tuple[float, float],
    ):
        input_shape = (23,)
        model = HardTanhModule(*activation_range)

        self.assert_delegated(model, input_shape, mocker)

    def test__unsupported_bounds(self):
        # TODO ONLY WHEN ALONE IN PARTITION
        input_shape = (2, 7, 2)
        min_value, max_value = float("-inf"), float("inf")
        model = HardTanhModule(min_value, max_value)

        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `hardtanh` was NOT delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [HardTanh])

    @pytest.mark.parametrize(
        "activation_range",
        [
            pytest.param((None, float("inf")), id="min = None, max = inf"),
            pytest.param((float("inf"), None), id="min = inf, max = None"),
        ],
    )
    def test__invalid_bounds(self, activation_range):
        # PyTorch doesn't allow these cases, so we cannot test our handling of this edge case.
        with pytest.raises(TypeError, match="'<=' not supported between instances of"):
            _ = HardTanhModule(*activation_range)

    @pytest.mark.parametrize(
        "min, max, expected_tflite_ops",
        [
            pytest.param(
                0.1,
                0.5,
                [Ops.ADD, Ops.MAXIMUM, Ops.MINIMUM],
                id="min = 0.1, max = 0.5 (Max/Min)",
            ),
            pytest.param(
                0.0, 1.0, [Ops.ADD, Ops.RELU_0_TO_1], id="min = 0, max = 1 (Relu0To1)"
            ),
            pytest.param(
                -1.0,
                1.0,
                [Ops.ADD, Ops.RELU_N1_TO_1],
                id="min = -1, max = 1 (ReluN1To1)",
            ),
            pytest.param(
                0.0,
                float("inf"),
                [Ops.ADD, Ops.RELU],
                id="min = 0, max = infinity (Relu)",
            ),
        ],
    )
    def test_convert_clamp__relu_vs_maxmin(self, mocker, min, max, expected_tflite_ops):
        input_shape = (23,)
        model = AddHardTanhModule(min, max)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
        neutron_ir_spy = mocker.spy(AtenModelBuilderDirector, "finish")

        delegated_ep = to_quantized_edge_program(
            model,
            input_shape,
        ).exported_program()

        # Make sure the `clamp` was delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
        assert not graph_contains_any_of_ops(delegated_ep.graph, [HardTanh])

        intermediate_ep = converter_spy.call_args.args[1]
        quant_node = list(intermediate_ep.graph.nodes)[-2]
        dequant_node = list(intermediate_ep.graph.nodes)[-4]
        tflite_internal_ops = [
            op.builtin_code for op in neutron_ir_spy.spy_return.operator_codes.vector
        ]

        assert graph_contains_any_of_ops(intermediate_ep.graph, [HardTanh])
        assert len(tflite_internal_ops) == len(expected_tflite_ops) + 1  # Transpose
        assert all(op in tflite_internal_ops for op in expected_tflite_ops)

        if len(expected_tflite_ops) == 3:
            # Min/Max variant should have same input and output quantization
            assert all(
                q == dq for q, dq in zip(quant_node.args[1:], dequant_node.args[1:])
            )
        else:
            assert not all(
                q == dq for q, dq in zip(quant_node.args[1:], dequant_node.args[1:])
            )
