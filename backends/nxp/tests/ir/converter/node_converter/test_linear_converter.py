# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare
from executorch.backends.nxp.tests.models import LinearModule
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_linear_conversion__with_bias():
    input_shape = (10, 32)
    edge_program = to_edge_program(
        LinearModule(bias=True), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    nodes = list(edge_program.graph.nodes)
    assert nodes[4].target == exir_ops.edge.aten.addmm.default
    assert len(nodes[4].args) == 3  # Has bias.

    convert_run_compare(edge_program, input_data=input_data)


def test_linear_conversion__without_bias():
    input_shape = (10, 32)
    edge_program = to_edge_program(
        LinearModule(bias=False), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    nodes = list(edge_program.graph.nodes)
    assert nodes[3].target == exir_ops.edge.aten.mm.default
    assert len(nodes[3].args) == 2  # No bias.

    convert_run_compare(edge_program, input_data=input_data)
