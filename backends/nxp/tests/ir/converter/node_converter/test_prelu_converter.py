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
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.models import (
    LinearPReLUModule,
    TwoPartitionPReLUModel,
)
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


# noinspection PyProtectedMember
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 8, 24, 32), id="4D."),
    ],
)
def test_prelu_with_linear_quant_conversion(mocker, input_shape):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    channels = input_shape[-1]
    edge_program = to_quantized_edge_program(
        LinearPReLUModule(in_features=channels, out_features=channels),
        input_shape,
    ).exported_program()

    # Capture generated entities
    neutron_ir_model, _ = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Check `prelu` was not decomposed into simpler edge operators
    assert not graph_contains_any_of_ops(
        exported_program.graph,
        [
            exir_ops.edge.aten.gt.Scalar,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.where.self,
        ],
    )

    assert graph_contains_any_of_ops(
        exported_program.graph,
        [exir_ops.edge.aten.prelu.default],
    )

    # Check `prelu` was delegated
    assert not graph_contains_any_of_ops(
        edge_program.graph,
        [exir_ops.edge.aten.prelu.default],
    )

    input_data = (
        (2 * np.random.random(input_shape).astype(np.float32) - 1) * 50
    ).astype(np.int8)

    convert_run_compare(exported_program, input_data, tfl_model=neutron_ir_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 8, 24, 32), id="4D."),
    ],
)
def test_prelu_2_partitions(mocker, input_shape):
    # TODO (Martin) Add a channels last dim order variant of this test to verify correct partitioning.
    # Run conversion
    edge_program = to_quantized_edge_program(
        TwoPartitionPReLUModel(), [input_shape, input_shape]
    ).exported_program()

    # Check `prelu` was delegated
    assert not graph_contains_any_of_ops(
        edge_program.graph,
        [exir_ops.edge.aten.prelu.default],
    )

    # Check there are two partitions
    edge_nodes = list(edge_program.graph.nodes)
    assert sum(n.target == ExecutorchDelegateCall for n in edge_nodes) == 2


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1,), id="1D not supported."),
        pytest.param((1, 8), id="2D not supported."),
        pytest.param((1, 8, 16), id="3D not supported."),
        pytest.param((1, 8, 16, 32, 64), id="5D not supported."),
        pytest.param((1, 8, 16, 31), id="channels must be divisible by NUM_MACS"),
        pytest.param((1, 8, 1024, 8), id="width*height is too big (limit 4096)"),
    ],
)
def test_prelu__no_delegation__unsupported_conversion(mocker, input_shape):
    # Run conversion
    channels = input_shape[-1]
    edge_program = to_quantized_edge_program(
        LinearPReLUModule(in_features=channels, out_features=channels),
        input_shape,
    ).exported_program()

    # Check `prelu` was not delegated (only `linear` was)
    edge_nodes = list(edge_program.graph.nodes)
    assert sum(n.target == ExecutorchDelegateCall for n in edge_nodes) == 1

    # Check `prelu` was decomposed into simpler edge operators
    assert graph_contains_any_of_ops(
        edge_program.graph,
        [
            exir_ops.edge.aten.gt.Scalar,
        ],
    )

    assert graph_contains_any_of_ops(
        edge_program.graph,
        [
            exir_ops.edge.aten.mul.Tensor,
        ],
    )

    assert graph_contains_any_of_ops(
        edge_program.graph,
        [
            exir_ops.edge.aten.where.self,
        ],
    )
