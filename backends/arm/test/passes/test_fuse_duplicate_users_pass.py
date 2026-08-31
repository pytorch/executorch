# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes import (
    EnsureUniqueOutputNodesPass,
    FuseDuplicateUsersPass,
    InsertRescalePass,
    RemoveNoopPass,
)
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.fx import Graph, GraphModule

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithOps(torch.nn.Module):
    ops_before_pass: Dict[str, int]
    ops_after_pass: Dict[str, int]


class FuseaAvgPool(ModuleWithOps):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 3,
    }
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1}

    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AvgPool2d(1)

    def forward(self, x):
        return self.avg(x) + self.avg(x) + self.avg(x)


class FuseAvgPoolChain(ModuleWithOps):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 6,
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


def _rescale_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    ]


def _graph_with_duplicate_rescale_users() -> GraphModule:
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(1, dtype=torch.int8))
    rescale_args = (x, torch.int32, [1.0], 16, 0)
    first_rescale = _set_val(
        graph.call_function(exir_ops.backend.tosa.RESCALE.default, rescale_args),
        torch.ones(1, dtype=torch.int32),
    )
    second_rescale = _set_val(
        graph.call_function(exir_ops.backend.tosa.RESCALE.default, rescale_args),
        torch.ones(1, dtype=torch.int32),
    )
    output = graph.output((first_rescale, second_rescale))
    output.meta["val"] = (
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    graph.lint()
    return GraphModule(torch.nn.Module(), graph)


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


def test_fuse_duplicate_users_keeps_identical_rescale_users():
    graph_module = _graph_with_duplicate_rescale_users()

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = FuseDuplicateUsersPass()(graph_module)

    rescale_nodes = _rescale_nodes(result.graph_module)

    result.graph_module.graph.lint()
    assert not result.modified
    assert len(rescale_nodes) == 2


def test_fuse_duplicate_users_keeps_nondeterministic_users():
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(2, 3))
    first = _set_val(
        graph.call_function(torch.ops.aten.rand_like.default, (x,)), torch.ones(2, 3)
    )
    second = _set_val(
        graph.call_function(torch.ops.aten.rand_like.default, (x,)), torch.ones(2, 3)
    )
    result = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (first, second)),
        torch.ones(2, 3),
    )
    graph.output(result)
    graph_module = GraphModule(torch.nn.Module(), graph)

    pass_result = FuseDuplicateUsersPass()(graph_module)

    random_nodes = [
        node
        for node in pass_result.graph_module.graph.nodes
        if node.target == torch.ops.aten.rand_like.default
    ]
    assert not pass_result.modified
    assert len(random_nodes) == 2


def test_fuse_duplicate_users_compares_literal_tensors_by_identity():
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(2, 3))
    first = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (x, torch.ones(2, 3))),
        torch.ones(2, 3),
    )
    second = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (x, torch.ones(2, 3))),
        torch.ones(2, 3),
    )
    result = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (first, second)),
        torch.ones(2, 3),
    )
    graph.output(result)
    graph_module = GraphModule(torch.nn.Module(), graph)

    pass_result = FuseDuplicateUsersPass()(graph_module)

    add_nodes = _add_node_names(pass_result.graph_module)
    assert not pass_result.modified
    assert len(add_nodes) == 3


def test_fuse_duplicate_users_respects_input_qparams():
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(2, 3))
    first = _set_val(
        graph.call_function(torch.ops.aten.view_copy.default, (x, [2, 3])),
        torch.ones(2, 3),
    )
    second = _set_val(
        graph.call_function(torch.ops.aten.view_copy.default, (x, [2, 3])),
        torch.ones(2, 3),
    )
    first.meta["input_qparams"] = {0: (0.25, 0)}
    second.meta["input_qparams"] = {0: (0.5, 0)}
    result = _set_val(
        graph.call_function(torch.ops.aten.add.Tensor, (first, second)),
        torch.ones(2, 3),
    )
    graph.output(result)
    graph_module = GraphModule(torch.nn.Module(), graph)

    pass_result = FuseDuplicateUsersPass()(graph_module)

    view_nodes = [
        node
        for node in pass_result.graph_module.graph.nodes
        if node.target == torch.ops.aten.view_copy.default
    ]
    assert not pass_result.modified
    assert len(view_nodes) == 2


def test_arm_fuse_duplicate_users_preserves_distinct_output_qparams():
    graph = Graph()
    x = _set_val(graph.placeholder("x"), torch.ones(2, 3))
    first = _set_val(
        graph.call_function(torch.ops.aten.neg.default, (x,)), torch.ones(2, 3)
    )
    second = _set_val(
        graph.call_function(torch.ops.aten.neg.default, (x,)), torch.ones(2, 3)
    )
    first.meta["output_qparams"] = {0: (0.25, 0)}
    second.meta["output_qparams"] = {0: (0.5, 0)}
    graph.output((first, second))
    graph_module = GraphModule(torch.nn.Module(), graph)

    pass_result = FuseDuplicateUsersPass(may_alias_outputs=True)(graph_module)

    neg_nodes = [
        node
        for node in pass_result.graph_module.graph.nodes
        if node.target == torch.ops.aten.neg.default
    ]
    assert not pass_result.modified
    assert len(neg_nodes) == 2


class LateDuplicateUsers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("first", torch.ones(2, 3))
        self.register_buffer("second", torch.ones(2, 3))

    def forward(self, x):
        return x + self.first, x + self.second


def test_fuse_duplicate_users_repairs_distinct_returned_values():
    exported_program = export(LateDuplicateUsers(), (torch.ones(2, 3),), strict=True)
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_exported_program = edge_program.exported_program()

    pass_manager = ArmPassManager(TosaCompileSpec("TOSA-1.0+FP"))
    graph_module = pass_manager.transform_to_backend_pipeline(
        edge_exported_program, edge_exported_program.graph_module
    )

    add_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.ADD.default
    ]
    identity_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.IDENTITY.default
    ]

    graph_module.graph.lint()
    assert len(add_nodes) == 1
    assert len(identity_nodes) == 2
    assert all(node.args == (add_nodes[0],) for node in identity_nodes)
    assert graph_module.graph.output_node().args[0] == tuple(identity_nodes)

    pass_types = [type(pass_) for pass_ in pass_manager.passes]
    post_noop_index = max(
        index
        for index, pass_type in enumerate(pass_types)
        if pass_type is RemoveNoopPass
    )
    assert pass_types[post_noop_index + 1 : post_noop_index + 4] == [
        FuseDuplicateUsersPass,
        InsertRescalePass,
        EnsureUniqueOutputNodesPass,
    ]
