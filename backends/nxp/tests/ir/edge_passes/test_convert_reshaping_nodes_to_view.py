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
from executorch.backends.nxp.tests.models import SqueezeAddModel, UnsqueezeAddModel
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


def _create_example_inputs(input_shape):
    """Helper function to create random int8 example inputs."""
    example_input_1 = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    example_input_2 = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    return {0: example_input_1, 1: example_input_2}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((2,), 0, id="1D."),
        pytest.param((8, 4, 6), 2, id="3D."),
        pytest.param((8, 4, 6, 8), -2, id="4D, negative dim."),
        pytest.param((8, 4, 6), 3, id="3D, dim arg is clipped."),
        pytest.param((8, 4, 6), -4, id="3D, dim arg is clipped."),
    ],
)
def test_convert_unsqueeze_to_view_full_pipeline(mocker, input_shape, dim):
    model = UnsqueezeAddModel(dim)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    full_delegated_program = to_quantized_edge_program(
        model,
        [input_shape, input_shape],
    ).exported_program()

    # Make sure no "aten.unsqueeze_copy.default" is in the model.
    assert not graph_contains_any_of_ops(
        full_delegated_program.graph,
        [
            exir_ops.edge.aten.unsqueeze_copy.default,
        ],
    )

    # Capture generated model
    neutron_ir_partition = converter_spy.spy_return[0]
    exported_program_partition: ExportedProgram = converter_spy.call_args.args[1]

    # Make sure "edge.aten.view_copy.default" is in the model that was converted to NeutronIR and delegated.
    assert graph_contains_any_of_ops(
        exported_program_partition.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    example_input = _create_example_inputs(input_shape)

    convert_run_compare(
        exported_program_partition,
        input_data=example_input,
        tfl_model=neutron_ir_partition,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((8, 1, 1), None, id="3D, dim = None."),
        pytest.param((8, 4, 1), 2, id="3D, dim hit."),
        pytest.param((8, 4, 1), 1, id="3D, dim miss."),
        pytest.param((8, 4, 1), -1, id="3D, negative dim hit."),
        pytest.param((8, 1, 1, 8), [1, 2], id="4D, full dims overlap."),
        pytest.param((8, 1, 4, 8), [1, 2], id="4D, partial dims overlap."),
        pytest.param((1, 8, 4, 8), [1, 2], id="4D, no dims overlap."),
        pytest.param((8, 1, 1, 8), [-2, -3], id="4D, negative full dims overlap."),
        pytest.param((8, 1, 4, 8), [-2, -3], id="4D, negative partial dims overlap."),
        pytest.param((1, 8, 4, 8), [-2, -3], id="4D, negative no dims overlap."),
        pytest.param(
            (8, 1, 1, 8), (1, 2), id="4D, tuple instead of list, full dims overlap."
        ),
        pytest.param(
            (8, 1, 4, 8), (1, 2), id="4D, tuple instead of list, partial dims overlap."
        ),
        pytest.param(
            (1, 8, 4, 8), (1, 2), id="4D, tuple instead of list, no dims overlap."
        ),
    ],
)
def test_convert_squeeze_to_view_full_pipeline(mocker, input_shape, dim):
    model = SqueezeAddModel(dim)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model,
        [input_shape, input_shape],
    ).exported_program()

    # Check that `Squeeze` is no longer present in the model
    assert not graph_contains_any_of_ops(
        edge_program.graph,
        [
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.aten.squeeze.default,
        ],
    )

    # Capture generated model
    neutron_ir_model = converter_spy.spy_return[0]
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Make sure `edge.aten.view_copy.default` is in the model.
    assert graph_contains_any_of_ops(
        exported_program.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    example_input = _create_example_inputs(input_shape)

    convert_run_compare(
        exported_program,
        input_data=example_input,
        tfl_model=neutron_ir_model,
    )
