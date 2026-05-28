# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.backends.transforms.utils import create_constant_placeholder
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Node


class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.aten.add.Tensor(x, y)


class AddAlphaModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.aten.add.Tensor(x, y, alpha=2)


def _count_target(graph_module: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target
    )


def _get_target_node(graph_module: torch.fx.GraphModule, target) -> Node:
    nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target
    ]
    assert len(nodes) == 1
    return nodes[0]


def _export_add_model() -> ExportedProgram:
    return torch.export.export(
        AddModel().eval(), (torch.randn(2, 3), torch.randn(2, 3)), strict=True
    )


def _export_add_alpha_model() -> ExportedProgram:
    return torch.export.export(
        AddAlphaModel().eval(), (torch.randn(2, 3), torch.randn(2, 3)), strict=True
    )


def test_rewrites_node_when_substitution_matches() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def replace_add_with_sub(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        del exported_program
        return DialectNodeSpec(torch.ops.aten.sub.Tensor, node.args)

    exported_program = _export_add_model()
    result = _TestAtenToDialectPass(exported_program=exported_program).call(
        exported_program.graph_module
    )

    assert result.modified
    assert _count_target(result.graph_module, torch.ops.aten.add.Tensor) == 0
    assert _count_target(result.graph_module, torch.ops.aten.sub.Tensor) == 1


def test_substitution_can_add_state_dict_placeholder() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def replace_add_rhs_with_constant(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        first_placeholder = next(
            graph_node
            for graph_node in node.graph.nodes
            if graph_node.op == "placeholder"
        )
        with node.graph.inserting_before(first_placeholder):
            const_node = create_constant_placeholder(
                exp_program=exported_program,
                graph=node.graph,
                name="test_constant",
                kind=InputKind.PARAMETER,
                data=torch.ones(2, 3),
            )
        return DialectNodeSpec(torch.ops.aten.add.Tensor, (node.args[0], const_node))

    exported_program = _export_add_model()
    result = _TestAtenToDialectPass(exported_program=exported_program).call(
        exported_program.graph_module
    )

    assert result.modified
    assert "test_constant" in exported_program.state_dict
    assert torch.equal(exported_program.state_dict["test_constant"], torch.ones(2, 3))
    assert (
        exported_program.graph_signature.inputs_to_parameters["test_constant"]
        == "test_constant"
    )
    add_node = _get_target_node(result.graph_module, torch.ops.aten.add.Tensor)
    assert add_node.args[1].name == "test_constant"

    x = torch.full((2, 3), 2.0)
    y = torch.full((2, 3), 5.0)
    torch.testing.assert_close(exported_program.module()(x, y), x + torch.ones_like(x))


def test_substitution_can_change_kwargs() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def replace_add_alpha(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        del exported_program
        return DialectNodeSpec(torch.ops.aten.add.Tensor, node.args, {"alpha": 3})

    exported_program = _export_add_alpha_model()
    result = _TestAtenToDialectPass(exported_program=exported_program).call(
        exported_program.graph_module
    )

    assert result.modified
    add_node = _get_target_node(result.graph_module, torch.ops.aten.add.Tensor)
    assert add_node.kwargs["alpha"] == 3

    x = torch.full((2, 3), 2.0)
    y = torch.full((2, 3), 5.0)
    torch.testing.assert_close(exported_program.module()(x, y), x + 3 * y)


def test_preserves_meta_when_substitution_matches() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def replace_add_with_sub(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        del exported_program
        return DialectNodeSpec(torch.ops.aten.sub.Tensor, node.args)

    exported_program = _export_add_model()
    add_node = _get_target_node(
        exported_program.graph_module, torch.ops.aten.add.Tensor
    )
    add_node.meta["test_sentinel"] = "kept"
    add_node.meta["stack_trace"] = "original stack"

    result = _TestAtenToDialectPass(exported_program=exported_program).call(
        exported_program.graph_module
    )

    sub_node = _get_target_node(result.graph_module, torch.ops.aten.sub.Tensor)
    assert sub_node.meta["test_sentinel"] == "kept"
    assert sub_node.meta["stack_trace"].startswith("original stack\n")
    assert sub_node.meta["stack_trace"] != "original stack"


def test_keeps_node_when_substitution_returns_none() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def do_not_replace(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        del node, exported_program
        return None

    exported_program = _export_add_model()
    result = _TestAtenToDialectPass(exported_program=exported_program).call(
        exported_program.graph_module
    )

    assert not result.modified
    assert _count_target(result.graph_module, torch.ops.aten.add.Tensor) == 1
    assert _count_target(result.graph_module, torch.ops.aten.sub.Tensor) == 0


def test_raises_when_duplicate_substitution_is_registered() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
    def first_replace(
        node: Node, exported_program: ExportedProgram
    ) -> DialectNodeSpec | None:
        del exported_program
        return DialectNodeSpec(torch.ops.aten.sub.Tensor, node.args)

    with pytest.raises(RuntimeError, match="Multiple substitutions registered"):

        @_TestAtenToDialectPass.register_dialect_substitution(torch.ops.aten.add.Tensor)
        def second_replace(
            node: Node, exported_program: ExportedProgram
        ) -> DialectNodeSpec | None:
            del exported_program
            return DialectNodeSpec(torch.ops.aten.mul.Tensor, node.args)


def test_ensures_raises_when_call_function_count_changes() -> None:
    class _TestAtenToDialectPass(AtenToDialectPass):
        pass

    exported_program = _export_add_model()
    graph_module = exported_program.graph_module
    test_pass = _TestAtenToDialectPass(exported_program=exported_program)
    test_pass.requires(graph_module)

    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    with graph_module.graph.inserting_before(output_node):
        graph_module.graph.create_node(
            "call_function",
            target=torch.ops.aten.sub.Tensor,
            args=tuple(placeholders),
            kwargs={},
        )

    with pytest.raises(RuntimeError, match="did not preserve"):
        test_pass.ensures(graph_module)
