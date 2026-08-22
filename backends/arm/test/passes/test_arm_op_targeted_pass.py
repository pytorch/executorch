# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.pass_base import ExportPass
from torch.fx import Graph, GraphModule
from torch.fx.passes.infra.pass_base import PassResult


TARGET_OP = torch.ops.aten.add.Tensor
OTHER_OP = operator.add


def create_graph_module(target=OTHER_OP, disallow_tfa: bool = False) -> GraphModule:
    graph = Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    lhs.meta["val"] = torch.randn(2, 3)
    rhs.meta["val"] = torch.randn(2, 3)
    node = graph.call_function(target, (lhs, rhs))
    node.meta["val"] = torch.randn(2, 3)
    if disallow_tfa:
        node.meta[DISALLOW_TFA_META_KEY] = True
    graph.output(node)
    return GraphModule(torch.nn.Module(), graph)


def create_test_pass_manager() -> ArmPassManager:
    compile_spec = TosaCompileSpec(
        TosaSpecification.create_from_string("TOSA-1.00+INT")
    )
    return ArmPassManager(compile_spec)


def run_single_pass(graph_module: GraphModule, test_pass: ExportPass) -> PassResult:
    pass_manager = create_test_pass_manager()
    pass_manager.add_pass(test_pass)
    return cast(PassResult, pass_manager(graph_module))


class DummyTargetedPass(ArmOpTargetedPass):
    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (TARGET_OP,)
    check_allowed_to_transform = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_operator_count = 0

    def call_operator(self, op, args, kwargs, meta):
        self.call_operator_count += 1
        return super().call_operator(op, args, kwargs, meta)


class InsertTargetPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        output = next(node for node in graph.nodes if node.op == "output")

        with graph.inserting_before(output):
            target_node = graph.call_function(
                TARGET_OP,
                (placeholders[0], placeholders[1]),
            )
            target_node.meta["val"] = torch.randn(2, 3)
        output.args = (target_node,)
        graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, True)


class CondModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg + 1

        def false_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg - 1

        return torch.cond(x.sum() > 0, true_branch, false_branch, [x])


def test_skips_when_target_is_absent() -> None:
    graph_module = create_graph_module()
    targeted_pass = DummyTargetedPass()

    result = run_single_pass(graph_module, targeted_pass)

    assert result is not None
    assert result.graph_module is graph_module
    assert not result.modified
    assert targeted_pass.call_operator_count == 0


def test_runs_when_target_is_present() -> None:
    graph_module = create_graph_module(TARGET_OP)
    targeted_pass = DummyTargetedPass()

    result = run_single_pass(graph_module, targeted_pass)

    assert result is not None
    assert result.modified
    assert targeted_pass.call_operator_count == 1


def test_skips_tfa_disallowed_target() -> None:
    graph_module = create_graph_module(TARGET_OP, disallow_tfa=True)
    targeted_pass = DummyTargetedPass(tfa_pass=True)

    result = run_single_pass(graph_module, targeted_pass)

    assert result is not None
    assert result.graph_module is graph_module
    assert not result.modified
    assert targeted_pass.call_operator_count == 0


def test_runs_when_previous_pass_creates_target() -> None:
    graph_module = create_graph_module()
    pass_manager = create_test_pass_manager()
    targeted_pass = DummyTargetedPass()
    pass_manager.add_pass(InsertTargetPass())
    pass_manager.add_pass(targeted_pass)
    result = pass_manager(graph_module)

    assert result.modified
    assert targeted_pass.call_operator_count == 1


def test_runs_when_target_is_present_in_nested_submodule() -> None:
    exported_program = torch.export.export(CondModule(), (torch.randn(2, 3),))
    graph_module = exported_program.graph_module
    targeted_pass = DummyTargetedPass()

    result = run_single_pass(graph_module, targeted_pass)

    assert result is not None
    assert result.modified
    assert targeted_pass.call_operator_count > 0
