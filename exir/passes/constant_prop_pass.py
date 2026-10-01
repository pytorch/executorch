# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import hashlib
import itertools
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, cast, Collection, Hashable, Mapping, Optional

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.operator.util import _QUANT_PRIMITIVES
from executorch.exir.passes.replace_aten_with_edge_pass import aten_to_edge
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_lifted_tensor_constant,
    is_param,
)
from torch._guards import detect_fake_mode
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.fx.node import map_aggregate
from torch.utils import _pytree as pytree

# Avoid propagating constants for `exir.ops.edge.aten.full.default`.
# Propagating aten.full can significantly increase compiled model size.
_DEFAULT_SKIP_TARGETS_NO_QUANT = {exir_ops.edge.aten.full.default}
_DEFAULT_SKIP_TARGETS = set(_DEFAULT_SKIP_TARGETS_NO_QUANT)

# Do not const prop quantization primitives
_QUANT_PRIMITIVES_EDGE = [aten_to_edge(op) for op in _QUANT_PRIMITIVES]
_DEFAULT_SKIP_TARGETS.update(set(_QUANT_PRIMITIVES_EDGE))


_PRIMITIVE_TYPES = (
    float,
    int,
    bool,
    str,
    torch.Tensor,
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
)

_PROP_TENSOR_CONSTANT_PREFIX = "_prop_tensor_constant"


def get_default_skip_targets_no_quant() -> set[EdgeOpOverload]:
    return _DEFAULT_SKIP_TARGETS_NO_QUANT


def is_const(
    arg,
    exported_program: ExportedProgram,
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
) -> bool:
    if isinstance(arg, (tuple, list)):
        return all(is_const(x, exported_program, const_node_to_tensor) for x in arg)
    elif isinstance(arg, dict):
        return all(
            is_const(x, exported_program, const_node_to_tensor) for x in arg.values()
        )
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return True
    elif arg is None:
        return True
    elif not isinstance(arg, torch.fx.Node):
        return False
    elif arg in const_node_to_tensor:
        return True
    return False


def get_data(
    arg,
    exported_program: ExportedProgram,
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
):
    if isinstance(arg, (tuple, list)):
        return type(arg)(
            get_data(x, exported_program, const_node_to_tensor) for x in arg
        )
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return arg
    elif arg in const_node_to_tensor:
        return const_node_to_tensor[arg]
    return None


def is_constant_buffer(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """Checks if the given node is a constant buffer."""

    if node.target not in program.graph_signature.inputs_to_buffers:
        return False
    fqn = program.graph_signature.inputs_to_buffers[node.target]
    # if the buffer is mutated then record that
    return fqn not in program.graph_signature.buffers_to_mutate.values()


def get_constant_placeholder_dict(
    exported_program: ExportedProgram,
    fold_buffers: bool = True,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Returns a dictionary of placeholder node -> constant tensor.
    """
    const_node_to_tensor: OrderedDict[torch.fx.Node, torch.Tensor] = OrderedDict()
    for node in exported_program.graph.find_nodes(op="placeholder"):
        if is_param(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_param(exported_program, node)
            )
        elif fold_buffers and is_constant_buffer(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_buffer(exported_program, node)
            )
        elif is_lifted_tensor_constant(exported_program, node):
            const_node_to_tensor[node] = cast(
                torch.Tensor, get_lifted_tensor_constant(exported_program, node)
            )
    return const_node_to_tensor


def custom_meta_key(node: torch.fx.Node) -> Hashable:
    """
    Returns the custom meta of `node`, such as the external file a weight is
    tagged for, in a form that compares and hashes by value; None when the
    node has none.
    """
    custom = node.meta.get("custom")
    key = tuple(sorted(custom.items())) if isinstance(custom, dict) else custom
    try:
        hash(key)
    except TypeError:
        return repr(key)
    return key


def get_propagated_const_tensor_dict(
    exported_program: ExportedProgram,
    custom_skip_targets: Optional[set[EdgeOpOverload]],
    fold_buffers: bool = True,
    nodes_to_fold: Optional[Collection[torch.fx.Node]] = None,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Propagates constants and returns a dictionary of node->constant tensors.
    """
    # Initialize dict with all constant placeholders.
    const_node_to_tensor = get_constant_placeholder_dict(exported_program, fold_buffers)
    # The custom meta of the placeholders each constant is computed from.
    source_customs = {
        node: frozenset({custom_meta_key(node)}) for node in const_node_to_tensor
    }

    if custom_skip_targets is not None:
        all_skip_targets = custom_skip_targets
    else:
        # Default set of targets to skip.
        all_skip_targets = _DEFAULT_SKIP_TARGETS

    for node in exported_program.graph.nodes:
        if (
            node.op != "call_function"
            or node.target is memory.alloc
            or node.target in all_skip_targets
            # Ops with side effects (RNG draws, mutation) have to run at
            # runtime. `aten.rand` has no tensor inputs, so without this check
            # it would be folded into a single frozen draw.
            or node.is_impure()
            # An allowlist restricts the fold to the nodes a caller has
            # chosen; a node outside it stays an op, and so do its consumers.
            or (nodes_to_fold is not None and node not in nodes_to_fold)
        ):
            continue

        if not is_const(
            node.args,
            exported_program,
            const_node_to_tensor,
        ) or not is_const(
            node.kwargs,
            exported_program,
            const_node_to_tensor,
        ):
            continue

        # A folded value is one tensor with one custom meta. Sources that
        # disagree on it, a weight and an adapter tagged for different
        # external files, stay apart, and so do the nodes computed from them.
        customs = frozenset().union(
            *(source_customs[input_node] for input_node in node.all_input_nodes)
        )
        if len(customs) > 1:
            continue
        source_customs[node] = customs

        args_data, kwargs_data = pytree.tree_map(
            lambda x: get_data(x, exported_program, const_node_to_tensor),
            (node.args, node.kwargs),
        )
        # Disable grad for constant propagation, otherwise the generated tensor can't be copied
        # because of the grad_fn.
        with torch.no_grad():
            # Execute the `node.target` and create a new propagated constant tensor.
            prop_constant_tensor = node.target(*args_data, **kwargs_data)

            # ExecuTorch doesn't support zero strides, so we need to ensure the tensor is contiguous
            # if it has any zero strides from broadcasting/expansion operations
            if (
                isinstance(prop_constant_tensor, torch.Tensor)
                and 0 in prop_constant_tensor.stride()
            ):
                prop_constant_tensor = prop_constant_tensor.contiguous()

        # Before decomposition a view op such as aten.t returns a view of the
        # parameter, and a view keeps requires_grad even under no_grad. A
        # later retrace clones such a constant outside no_grad, which leaves
        # a non-leaf tensor that cannot be deep-copied. Detach so the
        # constant is a plain leaf. A result that is not a tensor, such as
        # the Python float of aten.item, is recorded as it is: its consumers
        # fold with it, and create_constant_nodes_and_return_specs keeps the
        # op itself when a consumer outside the fold still needs the value.
        const_node_to_tensor[node] = pytree.tree_map(
            lambda leaf: leaf.detach() if isinstance(leaf, torch.Tensor) else leaf,
            prop_constant_tensor,
        )

    return const_node_to_tensor


def get_first_user_input(exported_program: ExportedProgram) -> torch.fx.Node:
    """Returns the first user input node in the graph."""
    first_user_input = None
    for node in exported_program.graph.nodes:
        if (
            node.op == "placeholder"
            and node.name in exported_program.graph_signature.user_inputs
        ):
            first_user_input = node
            break
    return first_user_input


def _source_placeholders(
    exported_program: ExportedProgram, node: torch.fx.Node
) -> list[torch.fx.Node]:
    """Returns the placeholders `node` is computed from, in graph order."""
    seen: set[torch.fx.Node] = set()
    stack = [node]
    while stack:
        for input_node in stack.pop().all_input_nodes:
            if input_node not in seen:
                seen.add(input_node)
                if input_node.op != "placeholder":
                    stack.append(input_node)
    return [
        placeholder
        for placeholder in exported_program.graph.find_nodes(op="placeholder")
        if placeholder in seen
    ]


def _source_spec(
    exported_program: ExportedProgram, source: torch.fx.Node
) -> tuple[Optional[str], InputKind]:
    """Returns the fully qualified name and the kind of a source placeholder."""
    signature = exported_program.graph_signature
    for mapping, kind in (
        (signature.inputs_to_parameters, InputKind.PARAMETER),
        (signature.inputs_to_buffers, InputKind.BUFFER),
        (signature.inputs_to_lifted_tensor_constants, InputKind.CONSTANT_TENSOR),
    ):
        if source.name in mapping:
            return mapping[source.name], kind
    return None, InputKind.CONSTANT_TENSOR


def _expression(exported_program: ExportedProgram, node: torch.fx.Node) -> str:
    """
    Returns a string that identifies the expression `node` computes.

    Every producer of `node` is described once, as a numbered definition
    that names its target fully qualified, `aten.t.default` or
    `_operator.getitem`, and its arguments; an argument that is a producer
    appears by its number and a placeholder by its fully qualified name. The
    same expression over the same parameters gives the same string in every
    method and export, whatever the node names are, and the string grows
    with the number of producers rather than with the number of paths
    through them. The walk is iterative, so a deep chain does not reach the
    recursion limit.
    """
    index: dict[torch.fx.Node, int] = {}
    definitions: list[str] = []

    def describe(arg):
        if not isinstance(arg, torch.fx.Node):
            return repr(arg)
        if arg.op == "placeholder":
            fqn, _ = _source_spec(exported_program, arg)
            return fqn if fqn is not None else arg.name
        return f"%{index[arg]}"

    # Post-order over the producers: a node is defined after its inputs.
    stack: list[tuple[torch.fx.Node, bool]] = [(node, False)]
    while stack:
        current, inputs_done = stack.pop()
        if current in index:
            continue
        if not inputs_done:
            stack.append((current, True))
            for input_node in reversed(current.all_input_nodes):
                if input_node.op != "placeholder" and input_node not in index:
                    stack.append((input_node, False))
            continue
        target = current.target
        if isinstance(target, EdgeOpOverload):
            name = target.__name__
        elif isinstance(target, torch._ops.OpOverload):
            name = str(target)
        else:
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", None)
            name = f"{module}.{qualname}" if module and qualname else repr(target)
        arguments = map_aggregate((current.args, current.kwargs), describe)
        index[current] = len(definitions)
        definitions.append(f"%{index[current]}={name}{arguments!r}")
    return ";".join(definitions)


def _folded_name(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
    source_fqn: Optional[str],
    taken: set[str],
    register_like_source: bool,
) -> str:
    """
    Returns the fully qualified name of a folded value.

    By default the name is `_prop_tensor_constant{N}`, numbered per program.
    With `register_like_source` it is the fully qualified name of the source
    the value is registered like, `_prop_`, and the whole sha256
    digest, 64 hex characters, of the expression that computes it: every
    method and export that folds the same expression over the same
    parameters produces the same name, and two expressions produce different
    ones. A tag function keyed on the name (the lora / foundation split of
    external weights) sees the source in it.
    """
    if register_like_source and source_fqn is not None:
        digest = hashlib.sha256(_expression(exported_program, node).encode())
        base = f"{source_fqn}_prop_{digest.hexdigest()}"
        candidates = (base if i == 0 else f"{base}_{i}" for i in itertools.count())
    else:
        prefix = _PROP_TENSOR_CONSTANT_PREFIX
        first = f"{prefix}{len(exported_program.constants) + len(taken)}"

        def bumped():
            yield first
            # The name is in use: continue past the largest suffix taken.
            suffix = max(
                (
                    int(name[len(prefix) :])
                    for name in (*exported_program.constants, *taken)
                    if name.startswith(prefix) and name[len(prefix) :].isdigit()
                ),
                default=-1,
            )
            while True:
                suffix += 1
                yield f"{prefix}{suffix}"

        candidates = bumped()
    signature = exported_program.graph_signature
    in_use = (
        taken
        | set(exported_program.constants)
        | set(exported_program.state_dict)
        | set(signature.inputs_to_parameters.values())
        | set(signature.inputs_to_buffers.values())
        | set(signature.inputs_to_lifted_tensor_constants.values())
    )
    return next(fqn for fqn in candidates if fqn not in in_use)


@dataclass
class _Fold:
    """A folded value, its placeholder and where it is registered."""

    placeholder: Optional[torch.fx.Node]
    fqn: str
    tensor: torch.Tensor
    spec: InputSpec

    @property
    def in_state_dict(self) -> bool:
        return self.spec.kind == InputKind.PARAMETER or (
            self.spec.kind == InputKind.BUFFER and bool(self.spec.persistent)
        )


def replace_with_constant_node(
    node: torch.fx.Node,
    prop_constant_tensor: torch.Tensor,
    insert_placeholder: Callable[[str, bool], torch.fx.Node],
    fake_mode,
    exported_program: ExportedProgram,
    taken: set[str],
    register_like_source: bool = False,
) -> _Fold:
    """
    Replaces `node` with a placeholder for `prop_constant_tensor`.

    The placeholder is created by `insert_placeholder`, given the name and
    whether the value belongs in the state dict; the value is not registered
    in the program here, the caller registers every fold once all
    placeholders exist.
    """
    source_fqn, kind = None, InputKind.CONSTANT_TENSOR
    source = None
    if register_like_source:
        # A parameter if any source is one, else a buffer if any source is
        # one, named after the first source of that kind in graph order.
        kinds = [InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR]
        source = min(
            _source_placeholders(exported_program, node),
            key=lambda p: kinds.index(_source_spec(exported_program, p)[1]),
            default=None,
        )
        if source is not None:
            source_fqn, kind = _source_spec(exported_program, source)
    fqn = _folded_name(exported_program, node, source_fqn, taken, register_like_source)
    taken.add(fqn)

    spec = InputSpec(
        kind=kind,
        arg=TensorArgument(name=""),
        target=fqn,
        persistent=None if kind == InputKind.PARAMETER else True,
    )
    fold = _Fold(None, fqn, prop_constant_tensor, spec)
    const_placeholder_node = insert_placeholder(fqn, fold.in_state_dict)
    fold.placeholder = const_placeholder_node
    spec.arg = TensorArgument(name=const_placeholder_node.name)

    # Update the meta data of the new placeholder node.
    for k, v in node.meta.items():
        const_placeholder_node.meta[k] = v
    if source is not None:
        # The custom meta of the sources, such as the external file a weight
        # is tagged for, describes the data, and the sources of a fold agree
        # on it. Carry it forward from them rather than from the arithmetic
        # node.
        const_placeholder_node.meta.pop("custom", None)
        if "custom" in source.meta:
            const_placeholder_node.meta["custom"] = dict(source.meta["custom"])
    const_placeholder_node.meta["val"] = fake_mode.from_tensor(
        prop_constant_tensor, static_shapes=True
    )
    const_placeholder_node.meta["val"].constant = prop_constant_tensor

    # Replace the original node with the new constant node.
    node.replace_all_uses_with(const_placeholder_node)
    exported_program.graph.erase_node(node)

    return fold


def _register_fold(exported_program: ExportedProgram, fold: _Fold) -> None:
    """
    Registers a folded value in the program under its kind.

    torch.export carries the meta of parameters and buffers through
    run_decompositions, but rebuilds lifted tensor constants without it, so a
    parameter's fold registered as a constant would lose the source's custom
    meta at the next decomposition.
    """
    if fold.spec.kind == InputKind.PARAMETER:
        exported_program.state_dict[fold.fqn] = torch.nn.Parameter(
            fold.tensor, requires_grad=False
        )
    elif fold.in_state_dict:
        exported_program.state_dict[fold.fqn] = fold.tensor
    else:
        exported_program.constants[fold.fqn] = fold.tensor


def _check_folds_registered_in_graph_order(
    exported_program: ExportedProgram,
    folds: list[_Fold],
    name_to_spec_dict: Mapping[str, InputSpec],
) -> None:
    """
    Checks that the placeholders of the folded values and their entries in
    the program agree: within the state dict and within the constants, the
    folds come after every value that was already there, in the order of
    their placeholders. A caller that binds the graph module by position,
    the state dict values then the constants then the user inputs, relies
    on it.
    """
    position = {
        node.name: i
        for i, node in enumerate(exported_program.graph.find_nodes(op="placeholder"))
    }
    folded = {fold.fqn for fold in folds}
    for store in (exported_program.state_dict, exported_program.constants):
        in_state_dict = store is exported_program.state_dict
        store_folds = sorted(
            (fold for fold in folds if fold.in_state_dict == in_state_dict),
            key=lambda fold: position[fold.placeholder.name],
        )
        registered = [fqn for fqn in store if fqn in folded]
        assert registered == [fold.fqn for fold in store_folds], (
            "constant_prop_pass registered folded values out of graph order: "
            f"{registered} in the program, {[f.fqn for f in store_folds]} in the graph"
        )
        if not store_folds:
            continue
        first_fold = position[store_folds[0].placeholder.name]
        for name, spec in name_to_spec_dict.items():
            if spec.target in store and name in position:
                assert position[name] < first_fold, (
                    f"constant_prop_pass placed a folded value before {spec.target}, "
                    "which was registered earlier"
                )


def get_fake_mode(exported_program: ExportedProgram):
    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )
    assert fake_mode is not None
    return fake_mode


def erase_constant_node(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
) -> None:
    # Remove corresponding tensor from param/constants dict.
    signature = exported_program.graph_signature
    if name := signature.inputs_to_parameters.get(node.name, None):
        exported_program.state_dict.pop(name, None)
    elif name := signature.inputs_to_lifted_tensor_constants.get(node.name, None):
        exported_program.constants.pop(name, None)
    elif name := signature.inputs_to_buffers.get(node.name, None):
        exported_program.constants.pop(name, None)
        exported_program.state_dict.pop(name, None)

    # Remove from graph.
    exported_program.graph.erase_node(node)


def create_constant_nodes_and_return_specs(
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
    exported_program: ExportedProgram,
    name_to_spec_dict: Mapping[str, InputSpec],
    register_like_source: bool = False,
) -> dict[str, InputSpec]:
    """
    Creates constant nodes for all entries in `const_node_to_tensor` and returns a node.name -> InputSpec dict.
    """
    new_specs: dict[str, InputSpec] = {}

    fake_mode = get_fake_mode(exported_program)
    graph = exported_program.graph
    first_user_input = get_first_user_input(exported_program)
    # Which store, the state dict (True) or the constants (False), holds the
    # value of each placeholder.
    store_of: dict[str, bool] = {}
    for name, spec in name_to_spec_dict.items():
        if spec.target in exported_program.state_dict:
            store_of[name] = True
        elif spec.target in exported_program.constants:
            store_of[name] = False

    def insert_placeholder(fqn: str, in_state_dict: bool) -> torch.fx.Node:
        # A folded placeholder goes after the last placeholder whose value is
        # in the same store, so that the placeholders of a store stay
        # together, in the order the store lists them. Without one it goes
        # before the first user input, or after the last placeholder.
        placeholders = graph.find_nodes(op="placeholder")
        anchor = next(
            (
                p
                for p in reversed(placeholders)
                if store_of.get(p.name) == in_state_dict
            ),
            None,
        )
        if anchor is not None:
            insert = graph.inserting_after(anchor)
        elif first_user_input is not None:
            insert = graph.inserting_before(first_user_input)
        else:
            insert = graph.inserting_after(placeholders[-1])
        with insert:
            placeholder = graph.placeholder(re.sub(r"[^0-9a-zA-Z_]+", "_", fqn))
        # The graph signature and the emitter look a placeholder up by its
        # node name, so the target has to be the name the graph settled on.
        placeholder.target = placeholder.name
        store_of[placeholder.name] = in_state_dict
        return placeholder

    folds: list[_Fold] = []
    taken: set[str] = set()
    # Constant nodes that stay ops: only a tensor can become a placeholder.
    kept: set[torch.fx.Node] = set()
    # Iterate over nodes in reverse order.
    for node, prop_constant_tensor in reversed(const_node_to_tensor.items()):
        if all(x in const_node_to_tensor and x not in kept for x in node.users):
            # All users of this constant node are also constant, so we don't need to create a new constant node.
            erase_constant_node(exported_program, node)
            continue

        if node.op == "placeholder":
            continue

        if not isinstance(prop_constant_tensor, torch.Tensor):
            # A Python scalar, the float of aten.item, or the tuple of a
            # multi-output op cannot be a placeholder. The consumers that
            # fold have folded by now; the op stays for the others, and the
            # nodes it is computed from stay with it.
            kept.add(node)
            continue

        fold = replace_with_constant_node(
            node,
            prop_constant_tensor,
            insert_placeholder,
            fake_mode,
            exported_program,
            taken,
            register_like_source,
        )
        folds.append(fold)
        new_specs[fold.placeholder.name] = fold.spec

    # Register the folded values once every placeholder exists, in graph
    # order, so that the program lists them the way the graph does.
    position = {node: i for i, node in enumerate(graph.find_nodes(op="placeholder"))}
    for fold in sorted(folds, key=lambda fold: position[fold.placeholder]):
        _register_fold(exported_program, fold)
    _check_folds_registered_in_graph_order(exported_program, folds, name_to_spec_dict)
    return new_specs


# add _skip_dim_order to ensure the introduced correct clone node for different dim order schema
# TODO(gasoonjia): only relying on _clone_dim_order once we remove _skip_dim_order option in the EdgeCompileConfig
def _update_output_node_and_specs(
    exported_program: ExportedProgram, _skip_dim_order: bool
) -> None:
    """
    Update the output node and output specs in the exported program.
    In case a constant node is used as output, we replace it with a clone of the constant node.
    """
    # Dict [node.name -> InputSpec]
    updated_constant_placeholders = get_constant_placeholder_dict(exported_program)
    output = exported_program.graph.find_nodes(op="output")[0]
    output_nodes = cast(list[torch.fx.Node], list(output.args[0]))
    output_specs = exported_program.graph_signature.output_specs
    assert len(output_nodes) == len(output_specs)

    clone_op = (
        exir_ops.edge.aten.clone.default
        if _skip_dim_order
        else exir_ops.edge.dim_order_ops._clone_dim_order.default
    )

    for i in range(len(output_specs)):
        out_node = output_nodes[i]
        if out_node not in updated_constant_placeholders:
            continue

        with exported_program.graph.inserting_after(out_node):
            new_node = exported_program.graph.call_function(clone_op, (out_node,))
        assert "val" in out_node.meta
        new_node.meta["val"] = out_node.meta["val"]
        output_nodes[i] = new_node

        # Update the constant-propagated output node.
        output_specs[i].arg = TensorArgument(name=output_nodes[i].name)

    output.args = (output_nodes,)


def constant_prop_pass(
    exported_program: ExportedProgram,
    custom_skip_targets: Optional[set[EdgeOpOverload]] = None,
    _skip_dim_order: bool = True,
    fold_buffers: bool = True,
    nodes_to_fold: Optional[Collection[torch.fx.Node]] = None,
    register_like_source: bool = False,
) -> ExportedProgram:
    """
    This pass is for constant propagation for Exported Program with lifted parameters,
    as the parameters will not be shown up as `get_attr` but as `placeholder` to the graph.

    Constants whose source placeholders disagree on their custom meta, such as a
    weight and an adapter tagged for different external files, are not folded
    together: each source keeps its data where it is tagged to go.

    Args:
        exported_program: The ExportedProgram to perform constant propagation on.
        custom_skip_targets: Optional set of EdgeOpOverload targets to skip during constant propagation.
        fold_buffers: Whether buffers this program does not mutate count as constants.
            The pass sees one method: a buffer this method only reads can be written
            by another method of the same program. Pass False to fold only parameters
            and lifted tensor constants.
        nodes_to_fold: Optional allowlist of the nodes to fold. Any other node stays
            an op, and so do the nodes computed from it.
        register_like_source: Whether a folded value is registered like its sources:
            a parameter if any of them is one, else a buffer if any is one, with
            the custom meta they share, and named after the first source of that
            kind and the expression, `w_prop_` and the 64-character sha256 digest
            of the fully qualified expression for a value computed from the
            parameter `w`. By default a folded value is a lifted tensor constant
            named `_prop_tensor_constant{N}`.

    Returns:
        The modified ExportedProgram with constant propagation applied.
    """
    if (
        len([node for node in exported_program.graph.nodes if node.op == "placeholder"])
        == 0
    ):
        return exported_program

    has_control_flow = [
        node
        for node in exported_program.graph.nodes
        if node.target == torch.ops.higher_order.cond
    ]
    if len(has_control_flow) > 0:
        logging.warning(
            "constant_prop_pass does not constant propagate in control flow modules"
        )

    const_node_to_tensor = get_propagated_const_tensor_dict(
        exported_program, custom_skip_targets, fold_buffers, nodes_to_fold
    )

    # Get old input specs.
    name_to_spec_dict = {
        s.arg.name: s for s in exported_program.graph_signature.input_specs
    }
    # Add the new constants to input specs dict.
    name_to_spec_dict.update(
        create_constant_nodes_and_return_specs(
            const_node_to_tensor,
            exported_program,
            name_to_spec_dict,
            register_like_source,
        )
    )

    # Generate new input spec.
    new_input_specs = []
    for node in exported_program.graph.find_nodes(op="placeholder"):
        new_input_specs.append(name_to_spec_dict[node.name])
    exported_program.graph_signature.input_specs = new_input_specs

    _update_output_node_and_specs(exported_program, _skip_dim_order=_skip_dim_order)

    # Cleanup the graph.
    exported_program.graph.eliminate_dead_code()
    exported_program.graph_module.recompile()

    return exported_program
