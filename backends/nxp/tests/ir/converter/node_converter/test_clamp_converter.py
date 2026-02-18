# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


# noinspection PyProtectedMember
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
Clamp = exir_ops.edge.aten.clamp.default


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


# noinspection PyShadowingBuiltins
@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(0, 6, id="min = 0, max = 6 (Relu6)"),
        pytest.param(0, 1, id="min = 0, max = 1 (Relu0To1)"),
        pytest.param(-1, 1, id="min = -1, max = 1 (ReluN1To1)"),
        pytest.param(0, None, id="min = 0, max = None (Relu)"),
        # float bounds.
        pytest.param(0.0, 6.0, id="min = 0.0, max = 6.0 (Relu6)"),
        pytest.param(0.0, 1.0, id="min = 0.0, max = 1.0 (Relu0To1)"),
        pytest.param(-1.0, 1.0, id="min = -1.0, max = 1.0 (ReluN1To1)"),
        pytest.param(0.0, None, id="min = 0.0, max = None (Relu)"),
    ],
)
def test_convert_clamp__supported(mocker, min, max):
    input_shape = (23,)
    model = AddClampModule(min, max)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `clamp` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Clamp])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `clamp`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [Clamp])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
    )


# noinspection PyShadowingBuiltins
@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(0, 6, id="min = 0, max = 6 (Relu6)"),
        pytest.param(0, None, id="min = 0, max = None (Relu)"),
    ],
)
def test_convert_clamp__single_op__not_delegated_variants(min, max):
    # Test that Clamp representable as Relu6 or Relu is NOT delegated, because it is a single op model which is not
    #  supported by Neutron.
    input_shape = (23,)
    model = ClampModule(min, max)

    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `clamp` was NOT delegated (single op model).
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [Clamp])


# noinspection PyShadowingBuiltins
@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(0, 1, id="min = 0, max = 1 (Relu0To1)"),
        pytest.param(-1, 1, id="min = -1, max = 1 (ReluN1To1)"),
    ],
)
def test_convert_clamp__single_op__delegated_variants(mocker, min, max):
    # Test that Clamp representable as Relu0To1 or ReluN1To1 is delegated, even though it is a single op model.
    input_shape = (23,)
    model = ClampModule(min, max)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `clamp` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Clamp])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `clamp`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [Clamp])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
    )


# noinspection PyShadowingBuiltins
@pytest.mark.parametrize(
    "min, max",
    [
        pytest.param(-3, 3, id="min = -3, max = 3"),
        pytest.param(None, 5, id="min = None, max = 5"),
    ],
)
def test_convert_clamp__no_delegation__unsupported_bounds(min, max):
    input_shape = (23,)
    model = AddClampModule(min, max)

    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `clamp` was NOT delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [Clamp])
