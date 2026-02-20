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


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
LeakyRelu2D = exir_ops.edge.aten.leaky_relu.default


def _assert_successful_delegation(model, input_shape, mocker, atol=0):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `leaky_relu` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [LeakyRelu2D])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `leaky_relu`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [LeakyRelu2D])

    convert_run_compare(
        intermediate_ep, tfl_model=neutron_ir_model, input_data=input_data, atol=atol
    )


class LeakyReluModule(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(*args, **kwargs)

    def forward(self, x):
        return self.leaky_relu(x)


@pytest.mark.parametrize(
    "alpha",
    [
        0.01,  # Default value.
        0.1,
        3.14159,
        0.0,
        1.0,
    ],
    ids=lambda alpha: f"alpha = {alpha}",
)
def test_convert_leaky_relu__alpha(mocker, alpha):
    _assert_successful_delegation(
        LeakyReluModule(negative_slope=alpha),
        (23,),
        mocker,
        atol=1,  # Common quantization rounding error.
    )


def test_convert_leaky_relu__default_alpha(mocker):
    _assert_successful_delegation(
        LeakyReluModule(),  # Leave the default alpha.
        (23,),
        mocker,
    )


@pytest.mark.parametrize(
    "inplace",
    [False, True],
    ids=lambda inplace: f"inplace = {inplace}",
)
def test_convert_leaky_relu__inplace(mocker, inplace):
    _assert_successful_delegation(
        LeakyReluModule(inplace=inplace),
        (23,),
        mocker,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (5,),
        (4, 5),
        (3, 4, 5),
        (2, 3, 4, 5),
        (1, 2, 3, 4, 5),
    ],
    ids=lambda input_shape: f"{len(input_shape)}D",
)
def test_convert_leaky_relu__ranks(mocker, input_shape: tuple[int, ...]):
    _assert_successful_delegation(
        LeakyReluModule(),
        input_shape,
        mocker,
        atol=1,  # Common quantization rounding error.
    )
