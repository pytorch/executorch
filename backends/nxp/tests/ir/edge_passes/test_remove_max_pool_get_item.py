# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.nxp.edge_passes.remove_max_pool_getitem_pass import (
    RemoveMaxPoolGetItemPass,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops

# noinspection PyProtectedMember
from executorch.exir.dialects._ops import ops as exir_ops


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, **kwargs)

    def forward(self, x):
        return self.max_pool2d(x)


def test_remove_max_pool_get_item_pass(mocker):
    model = MaxPool2dModule()
    input_shape = (1, 3, 12, 12)

    # Spy on the pass.
    spy = mocker.spy(RemoveMaxPoolGetItemPass, "run")

    edge_program = to_quantized_edge_program(
        model,
        input_shape,
    ).exported_program()

    # We cannot extract the graph before the pass, because it is modified inplace. So accessing the 2nd argument of the
    #  first call of the pass (which is the graph) returns the graph which is already modified by the pass.
    # But at least we can access the return value to determine if the pass made a modification.
    assert spy.spy_return_list[0].modified, "The pass did not modify the graph."

    # Make sure the `aten.max_pool2d_with_indices.default` and `getitem` were replaced by `aten.max_pool2d.default`.
    assert not graph_contains_any_of_ops(
        edge_program.graph,
        [exir_ops.edge.aten.max_pool2d_with_indices.default, operator.getitem],
    )
    assert graph_contains_any_of_ops(
        edge_program.graph,
        [
            exir_ops.edge.aten.max_pool2d.default,
        ],
    )
