# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Serialize a torch.fx graph into the native backend's generic flatbuffer format
(native_graph.fbs) and back.

The format is a topological list of fx nodes: each node carries an op-kind, a
target op-name string, and generic arguments encoded through a tagged Argument
union. Values are referenced by fx SSA name; tensor metadata lives in a side
table. Constant tensor data is NOT embedded here - it is returned separately so
the caller can resolve it at load time through an external fqn -> tensor map
(e.g. an ExecuTorch NamedDataStore/NamedDataMap for ET, or a custom store /
safetensors for the standalone native runtime), keyed by fqn.

Uses the "runtime flatc" pattern: the schema and the flatc binary are shipped as
package resources, and flatc is invoked to convert JSON <-> binary.
"""

import importlib.resources
import json
import os
import tempfile
from dataclasses import fields, is_dataclass
from enum import IntEnum
from typing import Any, cast, get_args, get_origin, get_type_hints, Union

import torch

from executorch.backends.native.serialization.schema import (
    Argument,
    ArgumentValue,
    BoolArg,
    BoolListArg,
    ConstantRef,
    FloatArg,
    FloatListArg,
    Graph,
    GraphArg,
    IntArg,
    IntListArg,
    InputKind as SchemaInputKind,
    Method,
    MutableBufferSpec,
    NamedArgument,
    Node,
    NoneArg,
    OpKind,
    OptionalTensorListArg,
    Output,
    OutputKind,
    OutputSpec,
    Program,
    ScalarType,
    ScalarTypeArg,
    StringArg,
    SymInt,
    SymIntArg,
    SymIntListArg,
    TensorArg,
    TensorListArg,
    TensorMeta,
    TensorValue,
)

from executorch.exir._serialize._dataclass import _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile

from torch.export.graph_signature import InputKind

# "major.minor": bump major for backward-incompatible schema changes (a field
# removed/reordered/retyped), minor for backward-compatible additions (append-only
# fields/union members/enum values). See the BC rules in native_graph.fbs.
SCHEMA_VERSION = "1.0"
_SCHEMA_RESOURCE = "native_graph.fbs"
_FILE_STEM = "native_graph"

_DTYPE_TO_SCALAR_TYPE: dict[torch.dtype, ScalarType] = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.float64: ScalarType.DOUBLE,
    torch.bool: ScalarType.BOOL,
    torch.bfloat16: ScalarType.BFLOAT16,
}
# Optional dtypes not present in every torch build.
for _name, _st in (
    ("uint16", ScalarType.UINT16),
    ("uint32", ScalarType.UINT32),
    ("uint64", ScalarType.UINT64),
):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _DTYPE_TO_SCALAR_TYPE[_dt] = _st


# ---------------------------------------------------------------------------
# Operator name (target) serialization - mirrors torch._export.serde.
# ---------------------------------------------------------------------------


def _resolve_op_overload(target: object) -> "torch._ops.OpOverload | None":
    """Return the aten OpOverload for an fx call_function target, or None.

    Edge-dialect ops (EdgeOpOverload) wrap the underlying aten OpOverload in
    ``_op``, so prefer that. A *plain* OpOverload also exposes ``_op``, but there
    it is the C++ builtin (empty ``__name__``) - unwrapping it loses the op name,
    so only unwrap when ``_op`` is itself an OpOverload and otherwise use the
    target directly. Non-OpOverload callables (sym builtins, ``operator.*``,
    higher-order ops) return None.
    """
    inner = getattr(target, "_op", None)
    if isinstance(inner, torch._ops.OpOverload):
        return inner
    if isinstance(target, torch._ops.OpOverload):
        return target
    return None


def serialize_operator(target: object) -> str:
    if isinstance(target, str):
        return target
    op = _resolve_op_overload(target)
    if op is not None:
        module = op.__module__.replace("torch._ops", "torch.ops")
        return f"{module}.{op.__name__}"
    # Fallback for non-OpOverload callables (e.g. operator.getitem, sym ops).
    module = getattr(target, "__module__", "") or ""
    name = getattr(target, "__name__", "") or str(target)
    module = module.replace("torch._ops", "torch.ops")
    return f"{module}.{name}" if module else name


# ---------------------------------------------------------------------------
# Metadata helpers.
# ---------------------------------------------------------------------------


def _scalar_type(dtype: torch.dtype) -> ScalarType:
    st = _DTYPE_TO_SCALAR_TYPE.get(dtype)
    if st is None:
        raise ValueError(f"Unsupported dtype for native serialization: {dtype}")
    return st


def _sym(x: object) -> SymInt:
    if isinstance(x, int):
        return SymInt(as_int=x)
    if isinstance(x, torch.SymInt):
        # maybe_as_int() returns an int only when the SymInt is genuinely
        # constant; it does NOT specialize a symbolic dim. (int(x) would guard to
        # the hint and silently freeze a dynamic shape to its example value.)
        concrete = x.node.maybe_as_int()
        if concrete is not None:
            return SymInt(as_int=concrete)
        return SymInt(as_symbol=str(x))
    return SymInt(as_symbol=str(x))


def _tensor_meta(t: torch.Tensor) -> TensorMeta:
    sizes = [_sym(s) for s in t.shape]
    try:
        strides = [_sym(s) for s in t.stride()]
    except RuntimeError:
        # Some tensors (e.g. sparse) do not expose strides; omit them.
        strides = []
    return TensorMeta(
        dtype=_scalar_type(t.dtype),
        sizes=sizes,
        strides=strides,
    )


# ---------------------------------------------------------------------------
# Argument dispatch.
# ---------------------------------------------------------------------------


def _to_list_arg(
    items: list[object], schema_type_hint: str | None = None
) -> ArgumentValue:
    if len(items) == 0:
        # Empty lists are ambiguous (IntList vs TensorList vs FloatList etc.).
        # If we have a schema type hint (e.g. "int[]", "Tensor[]"), use it to
        # emit a typed empty list. Otherwise fail loud so the caller must handle
        # the gap explicitly rather than silently coercing.
        hint = (schema_type_hint or "").lower()
        if "tensor?" in hint:
            return OptionalTensorListArg(names=[], has_value=[])
        if "tensor" in hint:
            return TensorListArg(names=[])
        if "bool" in hint:
            return BoolListArg(values=[])
        if "float" in hint or "double" in hint:
            return FloatListArg(values=[])
        if "symint" in hint or "sym_int" in hint:
            return SymIntListArg(values=[])
        if "int" in hint or hint == "":
            # int[] is the most common default (e.g. reduction dims), but still
            # require explicit handling for non-schema path: fail loud if no hint.
            if schema_type_hint is not None:
                return IntListArg(values=[])
        raise ValueError(
            f"Cannot serialize empty list argument without explicit element type: "
            f"empty list is ambiguous (e.g. Tensor[] vs int[]). "
            f"Schema hint: {schema_type_hint!r}, items: {items!r}. "
            f"If this is a schema default, ensure the op schema provides a typed "
            f"default or handle this case in _named_arguments."
        )
    if all(isinstance(x, torch.fx.Node) for x in items):
        return TensorListArg(names=[cast(torch.fx.Node, x).name for x in items])
    if all(x is None or isinstance(x, torch.fx.Node) for x in items):
        return OptionalTensorListArg(
            names=[cast(torch.fx.Node, x).name if x is not None else "" for x in items],
            has_value=[x is not None for x in items],
        )
    if all(isinstance(x, bool) for x in items):
        return BoolListArg(values=[cast(bool, x) for x in items])
    if all(isinstance(x, int) and not isinstance(x, bool) for x in items):
        return IntListArg(values=[cast(int, x) for x in items])
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
        return FloatListArg(values=[float(cast(float, x)) for x in items])
    # Symbolic/mixed int lists (e.g. dynamic view/expand sizes like [s0, -1]).
    # Symbolic dims arrive either as scalar SymInts or as references to in-graph
    # sym_size nodes; serialize each element as a SymInt (as_int for concrete dims,
    # as_symbol for a symbol or referenced value name) so they survive the round
    # trip instead of being dropped.
    if all(
        isinstance(x, (int, torch.SymInt, torch.fx.Node))
        and not isinstance(x, bool)
        for x in items
    ):
        values: list[SymInt] = []
        for x in items:
            if isinstance(x, torch.fx.Node):
                values.append(SymInt(as_symbol=x.name))
            else:
                values.append(_sym(x))
        return SymIntListArg(values=values)
    # No known variant fits (e.g. a mixed float/None/other list). Fail loud rather
    # than silently coercing to ints or dropping to an empty list -- an
    # unrepresentable arg means a real schema gap to handle explicitly.
    raise ValueError(
        f"Cannot serialize list argument with element types "
        f"{sorted({type(x).__name__ for x in items})}: {items!r}"
    )


def _to_arg_value(
    v: object,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
    schema_type_hint: str | None = None,
) -> ArgumentValue:
    if v is None:
        return NoneArg()
    if isinstance(v, torch.fx.Node):
        # A get_attr node that resolves to a submodule GraphModule is a
        # higher-order-op subgraph (torch.cond branch, map body, ...). Inline it
        # as a GraphArg rather than referencing it like a tensor value.
        if subgraph_map is not None and v.name in subgraph_map:
            sub, _ = _build_graph_body(subgraph_map[v.name], None, {}, None)
            return GraphArg(name=v.name, graph=sub)
        return TensorArg(name=v.name)
    if isinstance(v, bool):
        return BoolArg(value=v)
    if isinstance(v, int):
        return IntArg(value=v)
    if isinstance(v, float):
        return FloatArg(value=v)
    if isinstance(v, str):
        return StringArg(value=v)
    if isinstance(v, torch.dtype):
        return ScalarTypeArg(value=_scalar_type(v))
    if isinstance(v, torch.memory_format):
        return StringArg(value=str(v))
    if isinstance(v, torch.SymInt):
        return SymIntArg(value=_sym(v))
    if isinstance(v, (list, tuple)):
        return _to_list_arg(list(v), schema_type_hint=schema_type_hint)
    # device/layout are single-target/always-strided for this backend, so a string
    # is enough; anything else is an unhandled arg type -- fail loud rather than
    # emit a lossy repr.
    if isinstance(v, (torch.device, torch.layout)):
        return StringArg(value=str(v))
    raise ValueError(f"Cannot serialize argument of type {type(v).__name__}: {v!r}")


def _argument(
    v: object,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
    schema_type_hint: str | None = None,
) -> Argument:
    return Argument(value=_to_arg_value(v, subgraph_map, schema_type_hint=schema_type_hint))


def _named_arguments(
    node: torch.fx.Node,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
) -> list[NamedArgument]:
    op = _resolve_op_overload(node.target)
    if op is None:
        # No schema available (sym ops, operator.*, higher-order ops): serialize
        # args as given, resolving any subgraph references to inline GraphArgs.
        result = [
            NamedArgument(name=None, arg=_argument(a, subgraph_map)) for a in node.args
        ]
        result += [
            NamedArgument(name=k, arg=_argument(v, subgraph_map))
            for k, v in node.kwargs.items()
        ]
        return result

    # Materialize every schema argument, filling defaults for anything the call
    # omitted, so the serialized node fully specifies the op invocation without the
    # consumer needing to know the op's default values.
    result = []
    for sarg, value, present in _bound_schema_args(node, op):
        if not present:
            # Required arg not provided -- leave it out (invalid graph otherwise).
            continue
        # A written arg is one the op mutates in-place (schema Tensor(a!)); the op
        # schema is the source of truth (see comment on NamedArgument.mutated).
        mutated = sarg.alias_info is not None and sarg.alias_info.is_write
        # Pass schema type string (e.g. "int[]", "Tensor[]") for empty-list disambiguation.
        type_hint = str(getattr(sarg, "type", "") or "")
        result.append(
            NamedArgument(
                name=sarg.name,
                arg=_argument(value, subgraph_map, schema_type_hint=type_hint),
                mutated=mutated,
            )
        )
    return result


def _bound_schema_args(
    node: torch.fx.Node, op: "torch._ops.OpOverload"
) -> list[tuple[Any, object, bool]]:
    """Bind each schema argument to the value the call supplies.

    Returns one (schema_arg, value, present) triple per ``op._schema.arguments``
    position, in schema order. ``present`` is False for a required arg the call
    omitted (an invalid graph); ``value`` is then meaningless.
    """
    bound: list[tuple[Any, object, bool]] = []
    for i, sarg in enumerate(op._schema.arguments):
        if i < len(node.args):
            bound.append((sarg, node.args[i], True))
        elif sarg.name in node.kwargs:
            bound.append((sarg, node.kwargs[sarg.name], True))
        elif sarg.has_default_value():
            bound.append((sarg, sarg.default_value, True))
        else:
            bound.append((sarg, None, False))
    return bound


def _output_alias_of(node: torch.fx.Node) -> str | None:
    """SSA name of the input value this node's output shares storage with, or None.

    A view/alias op's return is annotated ``Tensor(a)`` (read-only view) or
    ``Tensor(a!)`` (in-place); the shared alias symbol matches one of the op's
    input args. We report the first return that aliases an in-graph input value,
    which covers the common single-return view and in-place ops. The write-ness of
    the sharing is carried by the aliased input's NamedArgument.mutated, so it is
    not repeated here.
    """
    op = _resolve_op_overload(node.target)
    if op is None:
        return None
    bound = _bound_schema_args(node, op)
    arg_sets = [
        (set(sarg.alias_info.before_set) if sarg.alias_info is not None else set())
        for sarg, _, _ in bound
    ]
    for ret in op._schema.returns:
        if ret.alias_info is None:
            continue
        ret_set = set(ret.alias_info.before_set)
        if not ret_set:
            continue
        for (_, value, present), aset in zip(bound, arg_sets):
            if present and (aset & ret_set) and isinstance(value, torch.fx.Node):
                return value.name
    return None


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def _op_kind(op: str) -> OpKind:
    try:
        return {
            "call_function": OpKind.CALL_FUNCTION,
            "placeholder": OpKind.PLACEHOLDER,
            "output": OpKind.OUTPUT,
        }[op]
    except KeyError:
        raise ValueError(f"Unsupported fx op {op!r}") from None


def _subgraph_map(
    graph_module: torch.fx.GraphModule,
) -> dict[str, torch.fx.GraphModule]:
    """Map get_attr node name -> submodule GraphModule for HOP subgraphs."""
    out: dict[str, torch.fx.GraphModule] = {}
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            attr = getattr(graph_module, str(node.target), None)
            if isinstance(attr, torch.fx.GraphModule):
                out[node.name] = attr
    return out


def _build_graph_body(
    graph_module: torch.fx.GraphModule,
    graph_signature: object | None,
    state_dict: dict[str, object],
    constants: dict[str, object] | None,
) -> tuple[Graph, dict[str, torch.Tensor]]:
    """Build a Graph from an fx GraphModule.

    ``graph_signature`` is the ExportGraphSignature for a top-level method, or
    ``None`` for a higher-order-op subgraph (which has no user-input/constant
    signature -- its placeholders are the operands passed by the enclosing op).
    Returns (graph, constant_data); constant_data is empty for subgraphs.
    """
    nodes: list[Node] = []
    subgraph_map = _subgraph_map(graph_module)

    # fx value name -> its tensor (from meta['val']). Captured for every node so
    # tensor_values includes inputs, outputs, mutable buffers, and all intermediates.
    val_by_name: dict[str, torch.Tensor] = {}

    def _all_tensor_values() -> list[TensorValue]:
        # Preserve topological order (insertion order of val_by_name).
        return [
            TensorValue(name=name, meta=_tensor_meta(tensor))
            for name, tensor in val_by_name.items()
        ]

    output_names: list[str] = []

    for node in graph_module.graph.nodes:
        # HOP subgraphs are inlined into the referencing arg as GraphArgs; drop the
        # get_attr node that names the submodule so it doesn't dangle.
        if node.op == "get_attr" and node.name in subgraph_map:
            continue

        # Any other get_attr is unserializable -- fail loud. HOP subgraph get_attrs
        # are inlined above, so a remaining get_attr means an UNLIFTED graph: in a
        # torch.export ExportedProgram, params/buffers/constants are lifted to
        # placeholders (see Graph.constants) and never appear as get_attr. Such a
        # node carries no fqn/data, so no runtime could resolve it. The serialized
        # format has no GET_ATTR op. (torch's own serde raises the same way.)
        if node.op == "get_attr":
            raise ValueError(
                f"get_attr node {node.name!r} (target {node.target!r}) cannot be "
                f"serialized: the graph is not lifted. Serialize a lifted "
                f"torch.export ExportedProgram (params/buffers/constants as inputs) "
                f"instead of an unlifted module."
            )

        kind = _op_kind(node.op)
        target = None
        inputs: list[NamedArgument] = []
        outputs: list[Output] = []

        if node.op == "output":
            out_vals = node.args[0] if node.args else ()
            if isinstance(out_vals, (list, tuple)):
                for v in out_vals:
                    inputs.append(NamedArgument(name=None, arg=_argument(v, subgraph_map)))
                    if isinstance(v, torch.fx.Node):
                        output_names.append(v.name)
            else:
                inputs.append(
                    NamedArgument(name=None, arg=_argument(out_vals, subgraph_map))
                )
                if isinstance(out_vals, torch.fx.Node):
                    output_names.append(out_vals.name)
        else:
            alias_of = None
            if node.op == "call_function":
                target = serialize_operator(node.target)
                inputs = _named_arguments(node, subgraph_map)
                alias_of = _output_alias_of(node)
            outputs = [Output(name=node.name, alias_of=alias_of)]

        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            val_by_name[node.name] = val

        dh = node.meta.get("debug_handle")
        nodes.append(
            Node(
                name=node.name,
                op_kind=kind,
                target=target,
                inputs=inputs or None,
                outputs=outputs or None,
                debug_handle=int(dh) if dh is not None else None,
            )
        )

    if graph_signature is None:
        # Subgraph: placeholders are the operands; no signature-derived data.
        user_inputs = [
            node.name for node in graph_module.graph.nodes if node.op == "placeholder"
        ]
        graph = Graph(
            nodes=nodes,
            inputs=user_inputs or None,
            outputs=output_names or None,
            tensor_values=_all_tensor_values() or None,
        )
        return graph, {}

    user_inputs = list(getattr(graph_signature, "user_inputs", []) or [])

    # Classify each graph output: real result vs a writeback into a persistent
    # buffer / user input. Keyed by the output value's SSA name.
    buffers_to_mutate = getattr(graph_signature, "buffers_to_mutate", None) or {}
    user_inputs_to_mutate = getattr(graph_signature, "user_inputs_to_mutate", None) or {}
    output_specs: list[OutputSpec] = []
    for name in output_names:
        if name in buffers_to_mutate:
            output_specs.append(
                OutputSpec(
                    name=name,
                    kind=OutputKind.BUFFER_MUTATION,
                    target=buffers_to_mutate[name],
                )
            )
        elif name in user_inputs_to_mutate:
            output_specs.append(
                OutputSpec(
                    name=name,
                    kind=OutputKind.USER_INPUT_MUTATION,
                    target=user_inputs_to_mutate[name],
                )
            )
        else:
            output_specs.append(OutputSpec(name=name, kind=OutputKind.USER_OUTPUT))

    # Buffers the graph mutates in-place (e.g. KV caches). Keyed by fqn so the
    # matching ConstantRef can be flagged as stateful rather than read-only.
    mutated_fqns = set(buffers_to_mutate.values())

    _INPUT_KIND_MAP = {
        InputKind.PARAMETER: SchemaInputKind.PARAMETER,
        InputKind.BUFFER: SchemaInputKind.BUFFER,
        InputKind.CONSTANT_TENSOR: SchemaInputKind.CONSTANT_TENSOR,
    }

    constant_refs: list[ConstantRef] = []
    constant_data: dict[str, torch.Tensor] = {}
    mutable_buffers: list[MutableBufferSpec] = []
    for ispec in getattr(graph_signature, "input_specs", []) or []:
        if ispec.kind not in _INPUT_KIND_MAP:
            continue
        arg = ispec.arg
        name = getattr(arg, "name", None)
        target_fqn = ispec.target
        if name is None or target_fqn is None:
            continue
        # Non-persistent buffer (e.g. a zero-initialized KV cache): graph state that
        # is zero-initialized at load. torch.export captures its current value in
        # `constants`, but it must NOT be shipped as data -- record it as a mutable
        # buffer (shape/dtype live in tensor_values; fqn is the cross-method sharing
        # identity). Checked before the data lookup precisely because its value is
        # present in `constants`.
        if ispec.kind == InputKind.BUFFER and getattr(ispec, "persistent", True) is False:
            mutable_buffers.append(MutableBufferSpec(name=name, fqn=target_fqn))
            continue
        tensor = None
        if target_fqn in state_dict:
            tensor = state_dict[target_fqn]
        elif constants is not None and target_fqn in constants:
            tensor = constants[target_fqn]
        if not isinstance(tensor, torch.Tensor):
            continue
        # Ship contiguous data, and compute meta from the same contiguous tensor so
        # the serialized strides match the bytes (a non-contiguous constant view
        # would otherwise record strides that disagree with the shipped layout).
        tensor = tensor.contiguous()
        constant_refs.append(
            ConstantRef(
                name=name,
                fqn=target_fqn,
                meta=_tensor_meta(tensor),
                kind=_INPUT_KIND_MAP[ispec.kind],
                mutated=target_fqn in mutated_fqns,
            )
        )
        constant_data[target_fqn] = tensor

    graph = Graph(
        nodes=nodes,
        inputs=user_inputs or None,
        outputs=output_names or None,
        tensor_values=_all_tensor_values() or None,
        constants=constant_refs or None,
        output_specs=output_specs or None,
        mutable_buffers=mutable_buffers or None,
    )
    return graph, constant_data


# ---------------------------------------------------------------------------
# JSON <-> flatbuffer via flatc.
# ---------------------------------------------------------------------------


def _encode(o: object) -> object:
    """Recursively convert a dataclass tree to JSON-compatible primitives.

    Emits the ``<field>_type`` discriminator BEFORE the union value (flatc
    requires the union type field to precede the value) and omits None-valued
    optional fields.
    """
    if is_dataclass(o):
        out: dict[str, object] = {}
        hints = get_type_hints(type(o))
        for f in fields(o):
            val = getattr(o, f.name)
            if val is None:
                continue
            hint = hints[f.name]
            if get_origin(hint) is Union and type(None) not in get_args(hint):
                out[f"{f.name}_type"] = type(val).__name__
                out[f.name] = _encode(val)
            else:
                out[f.name] = _encode(val)
        return out
    if isinstance(o, IntEnum):
        return int(o)
    if isinstance(o, (list, tuple)):
        return [_encode(x) for x in o]
    return o


def _prepare_schema(out_dir: str) -> str:
    data = importlib.resources.read_binary(__package__, _SCHEMA_RESOURCE)
    schema_path = os.path.join(out_dir, _SCHEMA_RESOURCE)
    with open(schema_path, "wb") as f:
        f.write(data)
    return schema_path


def _compile_to_bytes(root: object) -> bytes:
    json_str = json.dumps(_encode(root))
    with tempfile.TemporaryDirectory() as td:
        schema_path = _prepare_schema(td)
        json_path = os.path.join(td, _FILE_STEM + ".json")
        with open(json_path, "w") as f:
            f.write(json_str)
        _flatc_compile(td, schema_path, json_path)
        bin_path = os.path.join(td, _FILE_STEM + ".bin")
        with open(bin_path, "rb") as f:
            return f.read()


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    # Compare on CPU to handle potential device mismatches and ensure a
    # deterministic comparison.
    ca = a.detach().cpu()
    cb = b.detach().cpu()
    # torch.equal treats NaN != NaN, so for floating point we compare
    # element-wise with an explicit NaN-aware clause: two tensors with identical
    # NaN payloads should be considered equal for dedup purposes.
    if ca.dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16):
        eq = (ca == cb) | (torch.isnan(ca) & torch.isnan(cb))
        return bool(eq.all().item())
    return bool(torch.equal(ca, cb))


def serialize_program(
    methods: dict[
        str,
        tuple[
            torch.fx.GraphModule, object, dict[str, object], dict[str, object] | None
        ],
    ],
) -> tuple[bytes, dict[str, torch.Tensor]]:
    """Serialize one or more named methods into a native flatbuffer Program.

    ``methods`` maps a method name (e.g. "forward") to a
    ``(graph_module, graph_signature, state_dict, constants)`` tuple. Returns
    (flatbuffer_bytes, constant_data), where constant_data maps a fully-qualified
    name to the constant tensor, merged (deduped by fqn) across all methods. The
    caller ships constant_data through whatever fqn-keyed data store the target
    runtime uses (an ExecuTorch NamedDataStore for ET, or a custom store /
    safetensors for the standalone native runtime).

    Cross-method sharing is by fqn: bundling methods here asserts they come from a
    single model namespace, so an fqn is the same buffer/constant everywhere. That
    assertion is *validated* -- if two methods carry the same fqn with different
    data (constants) or different shape/dtype (mutable buffers), this raises rather
    than silently aliasing/clobbering.
    """
    method_objs: list[Method] = []
    constant_data: dict[str, torch.Tensor] = {}
    mutable_meta: dict[str, TensorMeta] = {}
    for name, (graph_module, graph_signature, state_dict, constants) in methods.items():
        graph, cdata = _build_graph_body(
            graph_module, graph_signature, state_dict, constants
        )
        method_objs.append(Method(name=name, graph=graph))

        for fqn, tensor in cdata.items():
            prev = constant_data.get(fqn)
            if prev is not None and not _same_tensor(prev, tensor):
                raise ValueError(
                    f"method {name!r}: constant fqn {fqn!r} conflicts with different "
                    f"data already provided by another method. Methods bundled into "
                    f"one program must share the same data per fqn."
                )
            constant_data[fqn] = tensor

        meta_by_name = {tv.name: tv.meta for tv in (graph.tensor_values or [])}
        for mb in graph.mutable_buffers or []:
            meta = meta_by_name.get(mb.name)
            if meta is None:
                continue
            prev_meta = mutable_meta.get(mb.fqn)
            if prev_meta is not None and prev_meta != meta:
                raise ValueError(
                    f"method {name!r}: mutable buffer fqn {mb.fqn!r} has a different "
                    f"shape/dtype than in another method; a shared buffer must match."
                )
            mutable_meta[mb.fqn] = meta

    program = Program(version=SCHEMA_VERSION, methods=method_objs)
    return _compile_to_bytes(program), constant_data


def serialize_graph(
    graph_module: torch.fx.GraphModule,
    graph_signature: object,
    state_dict: dict[str, object],
    constants: dict[str, object] | None = None,
) -> tuple[bytes, dict[str, torch.Tensor]]:
    """Serialize a single fx graph as a one-method ("forward") Program.

    Convenience wrapper around ``serialize_program``. Returns (flatbuffer_bytes,
    constant_data) as documented there.
    """
    return serialize_program(
        {"forward": (graph_module, graph_signature, state_dict, constants)}
    )


def deserialize_program(data: bytes) -> Program:
    """Deserialize native flatbuffer bytes back into a Program dataclass."""
    with tempfile.TemporaryDirectory() as td:
        schema_path = _prepare_schema(td)
        bin_path = os.path.join(td, _FILE_STEM + ".bin")
        with open(bin_path, "wb") as f:
            f.write(data)
        _flatc_decompile(td, schema_path, bin_path)
        json_path = os.path.join(td, _FILE_STEM + ".json")
        with open(json_path) as f:
            obj = json.load(f)
    return _json_to_dataclass(obj, Program)


def deserialize_graph(data: bytes) -> Graph:
    """Deserialize and return the first method's Graph (single-method convenience)."""
    program = deserialize_program(data)
    if not program.methods:
        raise ValueError("program has no methods")
    return program.methods[0].graph


def validate_graph(graph: Graph, available_data_keys: set[str] | None = None) -> None:
    """Assert the graph is self-contained: every value reference resolves and,
    when ``available_data_keys`` is given, every constant's data is present.

    This enforces that a ``.ptn`` (this graph) plus its ``.ptd`` (the named data,
    whose keys are ``available_data_keys``) carry everything needed to run --
    nothing extra is required from the ``.pte``. Raises ``ValueError`` on the
    first inconsistency.

    Checks: every ``TensorArg`` / tensor-list element references a defined value
    (a node output, a user input, a constant, or a mutable buffer); every output
    ``alias_of`` resolves; every input/output/mutable-buffer has tensor metadata;
    and every ``ConstantRef.fqn`` is available in the data keys. Mutable buffers
    are not data-backed, so they are exempt from the data-keys check.
    """
    defined: set[str] = {node.name for node in graph.nodes}
    defined.update(graph.inputs or [])
    defined.update(c.name for c in (graph.constants or []))
    defined.update(mb.name for mb in (graph.mutable_buffers or []))
    meta_names = {tv.name for tv in (graph.tensor_values or [])}

    def check_ref(name: str, ctx: str) -> None:
        if name and name not in defined:
            raise ValueError(f"{ctx}: unresolved value reference {name!r}")

    for node in graph.nodes:
        for out in node.outputs or []:
            if out.alias_of:
                check_ref(out.alias_of, f"node {node.name!r} output alias")
        for na in node.inputs or []:
            value = na.arg.value
            if isinstance(value, TensorArg):
                check_ref(value.name, f"node {node.name!r}")
            elif isinstance(value, TensorListArg):
                for n in value.names:
                    check_ref(n, f"node {node.name!r}")
            elif isinstance(value, OptionalTensorListArg):
                for n, present in zip(value.names, value.has_value):
                    if present:
                        check_ref(n, f"node {node.name!r}")
            elif isinstance(value, GraphArg):
                # HOP subgraph: self-contained (its own placeholder namespace).
                validate_graph(value.graph, available_data_keys)

    for name in graph.outputs or []:
        check_ref(name, "output")

    for name in graph.inputs or []:
        if name not in meta_names:
            raise ValueError(f"input {name!r} missing tensor metadata")
    for name in graph.outputs or []:
        if name not in meta_names:
            raise ValueError(f"output {name!r} missing tensor metadata")

    for mb in graph.mutable_buffers or []:
        if mb.name not in meta_names:
            raise ValueError(
                f"mutable buffer {mb.name!r} (fqn {mb.fqn!r}) missing tensor metadata"
            )

    if available_data_keys is not None:
        for c in graph.constants or []:
            if c.fqn not in available_data_keys:
                raise ValueError(
                    f"constant {c.name!r} (fqn {c.fqn!r}) has no data in the "
                    f"provided keys (.ptd/named-data)"
                )


def validate_program(
    program: Program, available_data_keys: set[str] | None = None
) -> None:
    """Validate every method's graph (see ``validate_graph``)."""
    for method in program.methods:
        validate_graph(method.graph, available_data_keys)
