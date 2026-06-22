# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from executorch.backends.arm._passes import FuseDuplicateUsersPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from torch.fx import Graph, GraphModule

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithOps(torch.nn.Module):
    ops_before_pass: Dict[str, int]
    ops_after_pass: Dict[str, int]


class FuseaAvgPool(ModuleWithOps):
    # CSE deduplicates the 3 identical avg(x) calls to 1 during to_edge
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1}

    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(1)

    def forward(self, x):
        return self.avg(x) + self.avg(x) + self.avg(x)


class FuseAvgPoolChain(ModuleWithOps):
    # CSE deduplicates the 3 identical avg(avg(x)) chains to 1 chain of 2 during to_edge
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 2,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 2}

    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(1)

    def forward(self, x):
        first = self.avg(self.avg(x))
        second = self.avg(self.avg(x))
        third = self.avg(self.avg(x))
        return first + second + third


modules: Dict[str, ModuleWithOps] = {
    "fuse_avg_pool": FuseaAvgPool(),
    "fuse_avg_pool_chain": FuseAvgPoolChain(),
}


def _set_val(node, val):
    node.meta["val"] = val
    return node


def _graph_with_users_not_in_node_order() -> GraphModule:
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(1))
    y = _set_val(graph.placeholder("y"), torch.ones(1))

    later_duplicate = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (x, y)), torch.ones(1)
    )
    with graph.inserting_before(later_duplicate):
        earlier_duplicate = _set_val(
            graph.call_function(torch.ops.aten.add.Tensor, (x, y)), torch.ones(1)
        )
        consumer = _set_val(
            graph.call_function(torch.ops.aten.neg.default, (earlier_duplicate,)),
            torch.ones(1),
        )

    output = graph.output(consumer)
    output.meta["val"] = torch.ones(1)
    graph.lint()
    return GraphModule(torch.nn.Module(), graph)


def _add_node_names(graph_module):
    return [
        node.name
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.add.Tensor
    ]


@common.parametrize("module", modules)
def test_fuse_duplicate_users_tosa_FP(module: ModuleWithOps):
    pipeline = PassPipeline[input_t](
        module=module,
        test_data=(torch.ones(1, 1, 1, 1),),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        pass_list=[
            FuseDuplicateUsersPass,
        ],
    )
    pipeline.run()


def test_fuse_duplicate_users_preserves_graph_order_for_representative():
    graph_module = _graph_with_users_not_in_node_order()
    assert _add_node_names(graph_module) == ["add_tensor_1", "add_tensor"]

    result = FuseDuplicateUsersPass()(graph_module)

    result.graph_module.graph.lint()
    assert result.modified
    assert len(_add_node_names(result.graph_module)) == 1
