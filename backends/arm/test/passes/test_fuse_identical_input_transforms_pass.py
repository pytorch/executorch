# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace
from typing import cast, Sequence

import pytest
import torch
from executorch.backends.arm import process_node
from executorch.backends.arm._passes.fuse_identical_input_transforms_pass import (
    FuseIdenticalInputTransformsPass,
    NormalizeTransformInputPlaceholdersPass,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult, ProxyValue
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.utils import _pytree as pytree


_ADD = exir_ops.edge.aten.add.Tensor
_CAT = exir_ops.edge.aten.cat.default
_VIEW = exir_ops.edge.aten.view_copy.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


def _count_node(
    graph_module: torch.fx.GraphModule, target: torch.fx.node.Target
) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _compute_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.node.Target]:
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def _call_function_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [node for node in graph_module.graph.nodes if node.op == "call_function"]


def _validate_numerics(
    original: torch.fx.GraphModule,
    modified: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    original.eval()
    modified.eval()
    with torch.no_grad():
        original_output = original(*inputs)
        modified_output = modified(*inputs)

    flat_original, _ = pytree.tree_flatten(original_output)
    flat_modified, _ = pytree.tree_flatten(modified_output)
    for original_tensor, modified_tensor in zip(flat_original, flat_modified):
        torch.testing.assert_close(original_tensor, modified_tensor)


def _apply_pass(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram | None = None,
) -> tuple[torch.fx.GraphModule, PassResult]:
    original = copy.deepcopy(graph_module)
    result = cast(
        PassResult,
        FuseIdenticalInputTransformsPass(exported_program).call(graph_module),
    )
    return original, result


def _binary_transform_graph(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
    y_transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
) -> torch.fx.GraphModule:
    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)

    x_node = _apply_transforms(builder, x, x_transforms)
    y_node = _apply_transforms(builder, y, y_transforms)
    add = builder.call_operator(op=_ADD, args=(x_node, y_node))
    builder.output([add])
    return builder.get_graph_module()


def _concat_transform_graph(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_transform: tuple[torch.fx.node.Target, Sequence[int]],
    y_transform: tuple[torch.fx.node.Target, Sequence[int]],
    dim: int,
) -> torch.fx.GraphModule:
    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)

    x_node = _apply_transforms(builder, x, [x_transform])
    y_node = _apply_transforms(builder, y, [y_transform])
    cat = builder.call_operator(op=_CAT, args=([x_node, y_node], dim))
    builder.output([cat])
    return builder.get_graph_module()


def _apply_transforms(
    builder: GraphBuilder,
    node: ProxyValue,
    transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
) -> ProxyValue:
    transformed = node
    for op, arg in transforms:
        transformed = builder.call_operator(op=op, args=(transformed, list(arg)))
    return transformed


def test_fuse_identical_input_transforms_sinks_view() -> None:
    x_data = torch.randn(6)
    y_data = torch.randn(6)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sunk_view_keeps_delegation_tag() -> None:
    x_data = torch.randn(6)
    y_data = torch.randn(6)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )
    add = next(
        node for node in _call_function_nodes(graph_module) if node.target == _ADD
    )
    add.meta["delegation_tag"] = "tag0"

    original, result = _apply_pass(graph_module)

    call_nodes = _call_function_nodes(result.graph_module)
    add = next(node for node in call_nodes if node.target == _ADD)
    view = next(node for node in call_nodes if node.target == _VIEW)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW]
    assert add.meta["delegation_tag"] == "tag0"
    assert view.meta["delegation_tag"] == "tag0"
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_permute() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sunk_permute_replaces_stale_tag() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 2, 1])],
    )
    for node in _call_function_nodes(graph_module):
        if node.target == _PERMUTE:
            node.meta["delegation_tag"] = "old_tag"
        if node.target == _ADD:
            node.meta["delegation_tag"] = "tag0"

    original, result = _apply_pass(graph_module)

    call_nodes = _call_function_nodes(result.graph_module)
    permute = next(node for node in call_nodes if node.target == _PERMUTE)

    assert result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE]
    assert permute.meta["delegation_tag"] == "tag0"
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_cat_input_views() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    cat_node = next(node for node in call_nodes if node.target == _CAT)
    view = next(node for node in call_nodes if node.target == _VIEW)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_CAT, _VIEW]
    assert [input_node.name for input_node in cat_node.args[0]] == ["x", "y"]
    assert cat_node.args[1] == 2
    assert cat_node.meta["val"].shape == torch.Size((2, 3, 8))
    assert view.args == (cat_node, [6, 8])
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sunk_cat_view_keeps_delegation_tag() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    cat.node.meta["delegation_tag"] = "tag0"
    builder.output([cat])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    call_nodes = _call_function_nodes(result.graph_module)
    view = next(node for node in call_nodes if node.target == _VIEW)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_CAT, _VIEW]
    assert view.meta["delegation_tag"] == "tag0"
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_preserves_transform_user_input_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.randn(2, 3)
    y.meta["val"] = torch.randn(2, 3)
    add = graph.call_function(_ADD, args=(x, y))
    add.meta["val"] = torch.randn(2, 3)
    graph.output((add,))
    graph_module = torch.fx.GraphModule({}, graph)
    x.name = "aten_view_copy_default"
    x.target = "aten_view_copy_default"
    y.name = "aten_view_copy_default_1"
    y.target = "aten_view_copy_default"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=x.name),
                        target=None,
                    ),
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=y.name),
                        target=None,
                    ),
                ],
                output_specs=[
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=add.name),
                        target=None,
                    )
                ],
            )
        ),
    )

    fuse_pass = FuseIdenticalInputTransformsPass()
    fuse_pass.exported_program = exported_program
    result = cast(
        PassResult,
        fuse_pass.call(graph_module),
    )

    placeholders = [
        node for node in result.graph_module.graph.nodes if node.op == "placeholder"
    ]
    input_spec_names = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
    ]

    monkeypatch.setattr(process_node, "process_inputs", lambda *args: None)
    for node in placeholders:
        process_node.process_placeholder(
            node,
            None,
            exported_program,
            None,
            cast(TosaSpecification, None),
        )

    assert result.modified
    assert [node.name for node in placeholders] == [
        "aten_view_copy_default",
        "aten_view_copy_default_1",
    ]
    assert [node.target for node in placeholders] == [
        "aten_view_copy_default",
        "aten_view_copy_default_1",
    ]
    assert input_spec_names == [
        "aten_view_copy_default",
        "aten_view_copy_default_1",
    ]


def test_fuse_identical_input_transforms_normalizes_multi_user_transform_input() -> (
    None
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.randn(2, 3)
    y.meta["val"] = torch.randn(2, 3)
    add = graph.call_function(_ADD, args=(x, y))
    sub = graph.call_function(exir_ops.edge.aten.sub.Tensor, args=(x, y))
    add.meta["val"] = torch.randn(2, 3)
    sub.meta["val"] = torch.randn(2, 3)
    graph.output((add, sub))
    graph_module = torch.fx.GraphModule({}, graph)
    x.name = "aten_view_copy_default_2"
    x.target = "aten_view_copy_default"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=x.name),
                        target=None,
                    ),
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=y.name),
                        target=None,
                    ),
                ],
                output_specs=[
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=add.name),
                        target=None,
                    ),
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=sub.name),
                        target=None,
                    ),
                ],
            )
        ),
    )

    _, result = _apply_pass(graph_module, exported_program)

    input_spec_names = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
    ]
    assert result.modified
    assert x.name == "aten_view_copy_default_2"
    assert x.target == "aten_view_copy_default_2"
    assert input_spec_names == ["aten_view_copy_default_2", "y"]


def test_normalize_transform_input_placeholders_does_not_sink_transforms() -> None:
    graph_module = _binary_transform_graph(
        torch.randn(6),
        torch.randn(6),
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )
    placeholder = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    placeholder.name = "aten_view_copy_default_2"
    placeholder.target = "aten_view_copy_default"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name="aten_view_copy_default"),
                        target=None,
                    )
                ],
                output_specs=[],
            )
        ),
    )

    result = NormalizeTransformInputPlaceholdersPass(exported_program).call(
        graph_module
    )

    input_spec_names = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
    ]
    assert result.modified
    assert placeholder.target == "aten_view_copy_default_2"
    assert input_spec_names == ["aten_view_copy_default_2"]
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]


def test_fuse_identical_input_transforms_normalizes_transform_input_signature() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.randn(2, 3)
    y.meta["val"] = torch.randn(2, 3)
    add = graph.call_function(_ADD, args=(x, y))
    add.meta["val"] = torch.randn(2, 3)
    graph.output((add,))
    graph_module = torch.fx.GraphModule({}, graph)
    x.name = "aten_view_copy_default_2"
    x.target = "aten_view_copy_default"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name="aten_view_copy_default"),
                        target=None,
                    ),
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=y.name),
                        target=None,
                    ),
                ],
                output_specs=[
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=add.name),
                        target=None,
                    )
                ],
            )
        ),
    )

    _, result = _apply_pass(graph_module, exported_program)

    input_spec_names = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
    ]
    assert result.modified
    assert x.target == "aten_view_copy_default_2"
    assert input_spec_names == ["aten_view_copy_default_2", "y"]


def test_fuse_identical_input_transforms_normalizes_non_user_transform_placeholder() -> (
    None
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.randn(2, 3)
    y.meta["val"] = torch.randn(2, 3)
    add = graph.call_function(_ADD, args=(x, y))
    add.meta["val"] = torch.randn(2, 3)
    graph.output((add,))
    graph_module = torch.fx.GraphModule({}, graph)
    x.name = "aten_view_copy_default_2"
    x.target = "aten_view_copy_default"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=y.name),
                        target=None,
                    ),
                ],
                output_specs=[
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=add.name),
                        target=None,
                    )
                ],
            )
        ),
    )

    _, result = _apply_pass(graph_module, exported_program)

    input_spec_names = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
    ]
    assert result.modified
    assert x.target == "aten_view_copy_default_2"
    assert input_spec_names == ["y"]


def test_fuse_identical_input_transforms_ignores_non_transform_placeholder_names() -> (
    None
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.randn(2, 3)
    y.meta["val"] = torch.randn(2, 3)
    add = graph.call_function(_ADD, args=(x, y))
    add.meta["val"] = torch.randn(2, 3)
    graph.output((add,))
    graph_module = torch.fx.GraphModule({}, graph)
    x.name = "custom_view_copy_default"
    x.target = "custom_view_copy_default_base"
    exported_program = cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=ExportGraphSignature(
                input_specs=[
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=x.name),
                        target=None,
                    ),
                    InputSpec(
                        kind=InputKind.USER_INPUT,
                        arg=TensorArgument(name=y.name),
                        target=None,
                    ),
                ],
                output_specs=[
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=add.name),
                        target=None,
                    )
                ],
            )
        ),
    )

    _, result = _apply_pass(graph_module, exported_program)

    assert not result.modified
    assert x.name == "custom_view_copy_default"
    assert x.target == "custom_view_copy_default_base"


def test_fuse_identical_input_transforms_rejects_cat_mixed_transforms() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _concat_transform_graph(
        x_data,
        y_data,
        (_VIEW, [2, 3, 4]),
        (_PERMUTE, [0, 1, 2]),
        0,
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_VIEW, _PERMUTE, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_cat_different_permutes() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 4, 3)
    graph_module = _concat_transform_graph(
        x_data,
        y_data,
        (_PERMUTE, [0, 1, 2]),
        (_PERMUTE, [0, 2, 1]),
        0,
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 2
    assert _compute_nodes(result.graph_module) == [_PERMUTE, _PERMUTE, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_shared_binary_input_transform() -> (
    None
):
    x_data = torch.randn(6)
    y_data = torch.randn(6)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [2, 3]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [2, 3]))
    add = builder.call_operator(op=_ADD, args=(x_view, y_view))
    builder.output([add, x_view])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_shared_cat_input_transform() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat, x_view])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_different_views() -> None:
    x_data = torch.randn(6)
    y_data = torch.randn(6)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [1, 2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_different_permutes() -> None:
    x_data = torch.randn(2, 1, 3)
    y_data = torch.randn(2, 1, 3)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 1, 2])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 2
    assert _compute_nodes(result.graph_module) == [_PERMUTE, _PERMUTE, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_view_permute_chain() -> None:
    x_data = torch.randn(6, 4)
    y_data = torch.randn(6, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3, 4]), (_PERMUTE, [0, 2, 1])],
        [(_VIEW, [2, 3, 4]), (_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_permute_view_chain() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1]), (_VIEW, [2, 2, 6])],
        [(_PERMUTE, [0, 2, 1]), (_VIEW, [2, 2, 6])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


@pytest.mark.parametrize(
    "x_shape, y_shape, view_shape",
    [
        ((1, 6), (6,), (2, 3)),
        ((1, 1, 6), (1, 6), (3, 2)),
    ],
)
def test_fuse_identical_input_transforms_sinks_views_through_rank_broadcasts(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    view_shape: tuple[int, ...],
) -> None:
    x_data = torch.randn(x_shape)
    y_data = torch.randn(y_shape)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, view_shape)],
        [(_VIEW, view_shape)],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_views_through_expanding_broadcast() -> (
    None
):
    x_data = torch.randn(1, 6)
    y_data = torch.randn(6, 1)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((1, 2, 3), (4, 2, 3)),
        ((4, 1, 3), (1, 2, 3)),
        ((4, 2, 1), (1, 1, 3)),
    ],
)
def test_fuse_identical_input_transforms_sinks_permutes_through_broadcasts(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
) -> None:
    x_data = torch.randn(x_shape)
    y_data = torch.randn(y_shape)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_cat_rejects_different_view_inputs() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(1, 6, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))
