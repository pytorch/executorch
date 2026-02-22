# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    ConvertUnsqueezeToViewPass,
    NeutronAtenPassManager,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)

from executorch.backends.nxp.tests.models import UnsqueezeAddModel
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


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
def test_convert_unsqueeze_to_view_simple(mocker, input_shape, dim):
    model = UnsqueezeAddModel(dim)

    example_input_1 = torch.rand(input_shape)
    example_input_2 = torch.rand(input_shape)

    exir_program_aten = torch.export.export(
        model,
        (example_input_1, example_input_2),
    ).module()

    # Check "aten.unsqueeze.default" is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.unsqueeze.default]
    )

    example_input = (example_input_1, example_input_2)
    outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [ConvertUnsqueezeToViewPass()])(
        exir_program_aten
    )

    # Make sure no "aten.unsqueeze.default" is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.unsqueeze.default],
    )

    # Make sure there is "aten.view.default" in the model.
    assert graph_contains_any_of_ops(
        exir_program_aten.graph,
        [torch.ops.aten.view.default],
    )

    outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)

    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


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
    edge_program = to_quantized_edge_program(
        model,
        [input_shape, input_shape],
    ).exported_program()

    # Make sure no "aten.unsqueeze.default" is in the model.
    assert not graph_contains_any_of_ops(
        edge_program.graph,
        [
            torch.ops.aten.unsqueeze.default,
        ],
    )

    # Capture generated model
    neutron_ir_model = converter_spy.spy_return[0]
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Make sure "edge.aten.view_copy.default" is in the model.
    assert graph_contains_any_of_ops(
        exported_program.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    example_input_1 = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    example_input_2 = (np.random.random(input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    example_input = {0: example_input_1, 1: example_input_2}

    convert_run_compare(
        exported_program,
        input_data=example_input,
        tfl_model=neutron_ir_model,
    )
