# Copyright 2026 NXP
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
from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    Clamp,
    ExecutorchDelegateCall,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa: F403 F401


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class ClampModule(torch.nn.Module):

    # noinspection PyShadowingBuiltins
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class AddClampModule(torch.nn.Module):

    # noinspection PyShadowingBuiltins
    def __init__(self, min=None, max=None):
        super().__init__()
        self.clamp = ClampModule(min, max)

    def forward(self, x):
        x = x + x
        return self.clamp(x)


class TestClamp:
    @pytest.mark.parametrize(
        "min, max",
        [
            pytest.param(-1, 2, id="min = -1, max = 2 (Max/Min)"),
            pytest.param(None, 1, id="min = None, max = 1 (Max/Min)"),
            pytest.param(1, None, id="min = 1, max = None (Max/Min)"),
            pytest.param(0, 2, id="min = 0, max = 2 (Max/Min)"),
            pytest.param(0, 1, id="min = 0, max = 1 (Relu0To1)"),
            pytest.param(-1, 1, id="min = -1, max = 1 (ReluN1To1)"),
            pytest.param(0, None, id="min = 0, max = None (Relu)"),
            # Float bounds
            pytest.param(-1.0, 2.0, id="min = -1.0, max = 2.0 (Max/Min)"),
            pytest.param(None, 1.0, id="min = None, max = 1.0 (Max/Min)"),
            pytest.param(1.0, None, id="min = 1.0, max = None (Max/Min)"),
            pytest.param(1.0, float("inf"), id="min = 1.0, max = infinity (Max/Min)"),
            pytest.param(-float("inf"), 1.0, id="min = infinity, max = 1.0 (Max/Min)"),
            pytest.param(0.1, 0.5, id="min = 0.1, max = 0.5 (Max/Min)"),
            pytest.param(0.0, 1.0, id="min = 0.0, max = 1.0 (Relu0To1)"),
            pytest.param(-1.0, 1.0, id="min = -1.0, max = 1.0 (ReluN1To1)"),
            pytest.param(0.0, None, id="min = 0, max = None (Relu)"),
        ],
    )
    def test_convert_clamp__full_pipeline(self, mocker, request, min, max, use_qat):
        input_shape = (2, 7, 2)  # Indivisible by num_macs
        model = AddClampModule(min, max)

        x_input_spec = ModelInputSpec(input_shape)
        comparator = NumericalStatsOutputComparator()
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={
                AddTensor: 1,
                Clamp: 1,
            },
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model=model,
            input_spec=[x_input_spec],
            dlg_model_verifier=graph_verifier,
            request=request,
            output_comparator=comparator,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "min, max",
        [
            pytest.param(
                float("inf"), float("inf"), id="min = inf, max = inf (invalid)"
            ),
            pytest.param(None, float("inf"), id="min = None, max = inf (invalid)"),
            pytest.param(float("inf"), None, id="min = inf, max = None (invalid)"),
        ],
    )
    def test_convert_clamp__invalid_bounds(self, min, max):
        input_shape = (2, 7, 2)
        model = ClampModule(min, max)

        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `clamp` was NOT delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [Clamp])

    # noinspection PyShadowingBuiltins
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
                0.0, None, [Ops.ADD, Ops.RELU], id="min = 0, max = None (Relu)"
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
        model = AddClampModule(min, max)

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
        tflite_spy = mocker.spy(AtenModelBuilderDirector, "finish")

        delegated_ep = to_quantized_edge_program(
            model,
            input_shape,
        ).exported_program()

        # Make sure the `clamp` was delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
        assert not graph_contains_any_of_ops(delegated_ep.graph, [Clamp])

        intermediate_ep = converter_spy.call_args.args[1]
        quant_node = list(intermediate_ep.graph.nodes)[-2]
        dequant_node = list(intermediate_ep.graph.nodes)[-4]
        tflite_internal_ops = [
            op.builtin_code for op in tflite_spy.spy_return.operator_codes.vector
        ]

        assert graph_contains_any_of_ops(intermediate_ep.graph, [Clamp])
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
