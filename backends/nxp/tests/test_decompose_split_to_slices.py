# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    DecomposeSplitToSlicesPass,
    NeutronAtenPassManager,
    SplitGRUBasedOnNumLayers,
)
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.executorch_pipeline import neutron_target_spec
from executorch.backends.nxp.tests.models import GRUModel, SplitWithSections, SplitWithSize
import numpy as np
    

@pytest.mark.parametrize(
    "input_shape, split_size, dim",
    [

        pytest.param((8,), 3, 0, id="1D."),
        pytest.param((4, 8), 5, 1, id="2D."),
    ],
)
def test_decompose_split_with_size(mocker, input_shape, split_size, dim):
    model = SplitWithSize(split_size, dim)
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    # Check "aten.split.Tensor" is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor]
    )
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [DecomposeSplitToSlicesPass()])(
        exir_program_aten
    )

    # Make sure no "Split" is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor, torch.ops.aten.split.default, torch.ops.aten.split_with_sizes.default]
    )
    
    # Check correct placement of slices
    nodes = list(exir_program_aten.graph.nodes)
    slices_count = input_shape[dim] // split_size
    # Slice nodes start appearing at index 1
    slices_start_idx = 1

    for i in range(0, slices_count):
        assert nodes[slices_start_idx + i].target == torch.ops.aten.slice_copy.Tensor

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


@pytest.mark.parametrize(
    "input_shape, sections, dim",
    [
        pytest.param((8,), [5, 3], 0, id="1D."),
        pytest.param((4, 8), [3, 3, 2], 1, id="2D."),
    ],
)
def test_decompose_split_with_section(mocker, input_shape, sections, dim):
    model = SplitWithSections(sections, dim)
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    # Check "aten.split_with_sizes" is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split_with_sizes.default]
    )
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [DecomposeSplitToSlicesPass()])(
        exir_program_aten
    )

    # Make sure no "Split" is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor, torch.ops.aten.split.default, torch.ops.aten.split_with_sizes.default]
    )

    # Check correct placement of slices
    nodes = list(exir_program_aten.graph.nodes)
    slices_count = len(sections)
    # Slice nodes start appearing at index 1
    slices_start_idx = 1

    for i in range(0, slices_count):
        assert nodes[slices_start_idx + i].target == torch.ops.aten.slice_copy.Tensor

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])
    

@pytest.mark.parametrize(
    "gru_layers",
    [
        pytest.param(2, id="2 GRU layers"),
    ],
)
def test_decompose_gru_with_split_default(mocker, gru_layers):
    model = GRUModel(gru_layers).eval()

    input_shape = (8, 1, 8)
    example_input = (torch.ones(input_shape),)

    exir_program_aten = torch.export.export(model, example_input).module()

    # Apply the pass to split the `aten.gru.input` into multiple instances, which adds a split operator
    NeutronAtenPassManager(neutron_target_spec, [SplitGRUBasedOnNumLayers()])(
        exir_program_aten
    )
    
    # Check "aten.split.default" is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.default]
    )
    
    outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [DecomposeSplitToSlicesPass()])(
        exir_program_aten
    )

    # Make sure no "Split" is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor, torch.ops.aten.split.default, torch.ops.aten.split_with_sizes.default]
    )

    # Check correct placement of slices
    nodes = list(exir_program_aten.graph.nodes)
    slices_count = gru_layers
    # Slice nodes start appearing at index 10 for gru_layer=2, for gru_layer=3 they start at index 14...
    slices_start_idx = 4 * gru_layers + 2

    for i in range(0, slices_count):
        assert nodes[slices_start_idx + i].target == torch.ops.aten.slice_copy.Tensor

    outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])


@pytest.mark.parametrize(
    "input_shape, size_or_sections, dim",
    [
        pytest.param((8, 4), 4, 1, id="2D, one chunk using split size."),
        pytest.param((8, 4), 5, 1, id="2D, one chunk using split size, chunk size over the limit."),
        pytest.param((8, 4), [4], 1, id="2D, one chunk using sections."),
    ],
)
def test_decompose_split_with_one_chunk(mocker, input_shape, size_or_sections, dim):
    if isinstance(size_or_sections, list):
        model = SplitWithSections(size_or_sections, dim)
    else:
        model = SplitWithSize(size_or_sections, dim)
    example_input = torch.rand(input_shape)

    exir_program_aten = torch.export.export(model, (example_input,)).module()

    # Check "aten.split" is present
    assert graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default]
    )
    outputs_before = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Apply the optimization.
    NeutronAtenPassManager(neutron_target_spec, [DecomposeSplitToSlicesPass()])(
        exir_program_aten
    )

    # Make sure no "Split" is in the model.
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.split.Tensor, torch.ops.aten.split.default, torch.ops.aten.split_with_sizes.default]
    )

    # Make sure there are no "aten.slice_copy.Tensor" either. Since the split was done using one chunk,
    # slicing is unnecessary
    assert not graph_contains_any_of_ops(
        exir_program_aten.graph, [torch.ops.aten.slice_copy.Tensor]
    )

    outputs_after = [o.detach().numpy() for o in exir_program_aten(example_input)]

    # Make sure the model still produces the exact same output.
    assert len(outputs_before) == len(outputs_after)
    
    for i in range(len(outputs_before)):
        assert np.allclose(outputs_before[i], outputs_after[i])
