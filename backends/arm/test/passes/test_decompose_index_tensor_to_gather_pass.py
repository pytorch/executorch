# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.decompose_index_tensor_to_gather_pass import (
    DecomposeIndexTensorToGatherPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from torch.export import export


class ConstantIndexTensor(torch.nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index: torch.Tensor
        self.register_buffer("index", torch.tensor([index], dtype=torch.int32))

    def forward(self, x: torch.Tensor):
        indices: list[slice | torch.Tensor] = [slice(None)] * x.dim()
        indices[self.dim] = self.index
        return x[tuple(indices)]


@pytest.mark.parametrize(
    "dim,index",
    (
        (0, 5),
        (1, -6),
    ),
)
def test_constant_out_of_bounds_index_raises(dim: int, index: int):
    exported_program = export(
        ConstantIndexTensor(dim, index),
        (torch.rand(5, 5, 5),),
    )
    edge_program = to_edge(exported_program)
    edge_exported_program = edge_program.exported_program()
    decompose_pass = DecomposeIndexTensorToGatherPass(
        edge_exported_program, decompose_constant_indices=True
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        with pytest.raises(
            IndexError,
            match=rf"index {index} is out of bounds for dimension {dim} with size 5",
        ):
            decompose_pass(edge_exported_program.graph_module)
