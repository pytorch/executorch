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
table. Constant tensor data is not embedded here; it is returned separately so the
caller can ship it separately in an external constant file, keyed by data_key.

Uses the "runtime flatc" pattern: the schema and the flatc binary are shipped as
package resources, and flatc is invoked to convert between JSON and binary.
"""

import importlib.resources
import json
import operator
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
    Dim,
    FloatArg,
    FloatListArg,
    Graph,
    GraphArg,
    InputKind as SchemaInputKind,
    IntArg,
    IntListArg,
    Method,
    MutableBufferSpec,
    NamedArgument,
    NamedTensorRef,
    Node,
    NoneArg,
    OpKind,
    OptionalTensorListArg,
    Output,
    OutputKind,
    OutputSpec,
    OutputValueKind,
    Program,
    ScalarType,
    ScalarTypeArg,
    StringArg,
    TensorArg,
    TensorListArg,
    TensorMeta,
    TensorValue,
)

from executorch.exir._serialize._dataclass import _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.exir.tensor import dim_order_from_stride, stride_from_dim_order

from torch.export.graph_signature import InputKind, TensorArgument
from torch.fx.experimental.symbolic_shapes import statically_known_true

SCHEMA_VERSION = "1"
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
# Operator name (target) serialization, mirroring torch._export.serde.
# ---------------------------------------------------------------------------


def _resolve_op_overload(target: object) -> "torch._ops.OpOverload | None":
    """Return the aten OpOverload for an fx call_function target, or None.

    Edge-dialect ops (EdgeOpOverload) wrap the underlying aten OpOverload in _op, so
    prefer that. A plain OpOverload also exposes _op, but there it is the C++ builtin
    (empty __name__); unwrapping it loses the op name, so only unwrap when _op is
    itself an OpOverload, otherwise use the target directly. Non-OpOverload callables
    (sym builtins, operator.*, higher-order ops) return None.
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


def _dim(x: object) -> Dim:
    if isinstance(x, int):
        return Dim(min=x, max=x)
    if isinstance(x, torch.SymInt):
        # maybe_as_int() returns an int only when the SymInt is genuinely constant; it
        # does not specialize a symbolic dim (int(x) would guard to the hint and freeze
        # a dynamic shape to its example value).
        concrete = x.node.maybe_as_int()
        if concrete is not None:
            return Dim(min=concrete, max=concrete)
        # Symbolic dim: record its value range from the ShapeEnv rather than a symbol
        # name (the runtime uses concrete shapes; symbols would need a C++ symbolic
        # engine). Unbounded above is max = -1.
        lower, upper = 0, -1
        try:
            vr = x.node.shape_env.bound_sympy(x.node.expr)
            if vr.lower.is_finite:
                lower = int(vr.lower)
            if vr.upper.is_finite:
                upper = int(vr.upper)
        except AttributeError:
            # No shape_env/expr/bound available; leave unbounded.
            pass
        except (ValueError, TypeError, RuntimeError):
            # Sympy bound computation failed; leave unbounded.
            pass
        return Dim(min=lower, max=upper)
    return Dim(min=int(x), max=int(x))


def _dim_order(t: torch.Tensor) -> list[int]:
    """Return axes sorted outer-to-inner by stride for a dim-order expressible tensor.

    Only dim_order is serialized, not raw strides, so the strides must be recoverable
    from sizes and dim_order. Raise if they are not (for example sliced, as_strided,
    broadcast, or overlapping layouts) instead of silently losing the layout. Symbolic
    strides are supported when they match the strides implied by the (possibly
    symbolic) sizes.
    """
    ndim = t.dim()
    if ndim == 0:
        return []
    strides = tuple(t.stride())
    sizes = list(t.shape)
    # dim_order_from_stride handles symbolic strides and rejects stride-0 layouts.
    dim_order = [int(d) for d in dim_order_from_stride(strides)]
    expected = stride_from_dim_order(sizes, dim_order)
    for i in range(ndim):
        # A size-1 dim only ever indexes 0, so its stride is arbitrary and need not
        # match the reconstruction (e.g. channels-last with a size-1 channel).
        if statically_known_true(sizes[i] == 1):
            continue
        # statically_known_true proves equality without adding shape guards.
        if not statically_known_true(strides[i] == expected[i]):
            raise ValueError(
                f"tensor is not dim-order expressible: strides {strides} with sizes "
                f"{tuple(sizes)} imply dim_order {tuple(dim_order)}, which reconstructs "
                f"strides {tuple(expected)}. Only dim_order is serialized, so this "
                f"layout (e.g. sliced/as_strided or overlapping) would be lost."
            )
    return dim_order


def _tensor_meta(t: torch.Tensor) -> TensorMeta:
    return TensorMeta(
        dtype=_scalar_type(t.dtype),
        sizes=[_dim(s) for s in t.shape],
        dim_order=_dim_order(t),
    )


# ---------------------------------------------------------------------------
# Argument dispatch.
# ---------------------------------------------------------------------------


def _int_val_node(x: object) -> bool:
    """True if x is an fx node whose produced value is an int / SymInt (e.g. a
    sym_size or shape-arithmetic node). Such a node referenced as an arg is a
    dynamic int (an IntArg carrying a ref), not a tensor."""
    if not isinstance(x, torch.fx.Node):
        return False
    val = x.meta.get("val")
    return isinstance(val, (int, torch.SymInt)) and not isinstance(val, bool)


def _is_tensor_node(x: object) -> bool:
    return isinstance(x, torch.fx.Node) and not _int_val_node(x)


def _is_optional_tensor_node(x: object) -> bool:
    return x is None or _is_tensor_node(x)


def _is_dynamic_int_element(x: object) -> bool:
    return (
        (isinstance(x, int) and not isinstance(x, bool))
        or isinstance(x, torch.SymInt)
        or _int_val_node(x)
    )


def _check_no_subgraph_in_list(
    items: list[object], subgraph_map: dict[str, torch.fx.GraphModule] | None
) -> None:
    if subgraph_map is None:
        return
    for x in items:
        if isinstance(x, torch.fx.Node) and x.name in subgraph_map:
            raise ValueError(
                f"higher-order-op subgraph {x.name!r} appears inside a list "
                f"argument; only a direct subgraph arg can be inlined as a "
                f"GraphArg. This graph shape is not supported."
            )


def _empty_typed_list_arg(
    schema_type_hint: str | None, items: list[object]
) -> ArgumentValue:
    hint = (schema_type_hint or "").lower()
    if "tensor?" in hint:
        return OptionalTensorListArg(names=[], has_value=[])
    if "tensor" in hint:
        return TensorListArg(names=[])
    if "bool" in hint:
        return BoolListArg(values=[])
    if "float" in hint or "double" in hint:
        return FloatListArg(values=[])
    if "int" in hint or hint == "":
        if schema_type_hint is not None:
            return IntListArg(values=[])
    raise ValueError(
        f"Cannot serialize empty list argument without explicit element type: "
        f"empty list is ambiguous (e.g. Tensor[] vs int[]). "
        f"Schema hint: {schema_type_hint!r}, items: {items!r}. "
        f"If this is a schema default, ensure the op schema provides a typed "
        f"default or handle this case in _named_arguments."
    )


def _dynamic_int_list_arg(items: list[object]) -> IntListArg:
    values: list[int] = []
    refs: list[str] = []
    for x in items:
        if isinstance(x, torch.fx.Node):
            values.append(0)
            refs.append(x.name)
        elif isinstance(x, torch.SymInt):
            concrete = x.node.maybe_as_int()
            if concrete is None:
                raise ValueError(
                    f"symbolic SymInt list element {x!r} is not backed by an fx "
                    f"node; dynamic int args must reference an in-graph value."
                )
            values.append(concrete)
            refs.append("")
        else:
            values.append(cast(int, x))
            refs.append("")
    return IntListArg(values=values, refs=refs)


def _to_list_arg(
    items: list[object],
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
    schema_type_hint: str | None = None,
) -> ArgumentValue:
    if len(items) == 0:
        return _empty_typed_list_arg(schema_type_hint, items)
    _check_no_subgraph_in_list(items, subgraph_map)
    if all(_is_tensor_node(x) for x in items):
        return TensorListArg(names=[cast(torch.fx.Node, x).name for x in items])
    if all(_is_optional_tensor_node(x) for x in items):
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
    if all(_is_dynamic_int_element(x) for x in items):
        return _dynamic_int_list_arg(items)
    raise ValueError(
        f"Cannot serialize list argument with element types "
        f"{sorted({type(x).__name__ for x in items})}: {items!r}"
    )


def _node_to_arg_value(
    v: torch.fx.Node,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None",
) -> ArgumentValue:
    # A get_attr node that resolves to a submodule GraphModule is a
    # higher-order-op subgraph (torch.cond branch, map body, ...). Inline it
    # as a GraphArg rather than referencing it like a tensor value.
    if subgraph_map is not None and v.name in subgraph_map:
        sub = _build_graph_body(subgraph_map[v.name], None, {}, None)[0]
        return GraphArg(name=v.name, graph=sub)
    # An int-producing node (sym_size / arith on shapes) referenced as an arg is a
    # dynamic int, not a tensor; serialize it as an IntArg carrying a ref.
    if _int_val_node(v):
        return IntArg(value=0, ref=v.name)
    # A bool/float-producing node (e.g. a shape comparison used as a cond predicate)
    # is a dynamic scalar referenced by SSA name, not a tensor.
    val = v.meta.get("val")
    if isinstance(val, (bool, torch.SymBool)):
        return BoolArg(value=False, ref=v.name)
    if isinstance(val, (float, torch.SymFloat)):
        return FloatArg(value=0.0, ref=v.name)
    return TensorArg(name=v.name)


def _scalar_to_arg_value(v: object) -> ArgumentValue:
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
    # device/layout/memory_format are single-target and always-strided for this
    # backend, so a string is enough.
    if isinstance(v, (torch.memory_format, torch.device, torch.layout)):
        return StringArg(value=str(v))
    if isinstance(v, torch.SymInt):
        # A raw SymInt arg is only representable if it is a constant; a genuinely
        # symbolic one must reference an in-graph value (an fx node), handled by
        # _node_to_arg_value.
        concrete = v.node.maybe_as_int()
        if concrete is not None:
            return IntArg(value=concrete)
        raise ValueError(
            f"symbolic SymInt scalar arg {v!r} is not backed by an fx node; "
            f"dynamic int args must reference an in-graph value."
        )
    # Anything else is an unhandled arg type, so fail loud rather than emit a lossy
    # repr.
    raise ValueError(f"Cannot serialize argument of type {type(v).__name__}: {v!r}")


def _to_arg_value(
    v: object,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
    schema_type_hint: str | None = None,
) -> ArgumentValue:
    if v is None:
        return NoneArg()
    if isinstance(v, torch.fx.Node):
        return _node_to_arg_value(v, subgraph_map)
    if isinstance(v, (list, tuple)):
        return _to_list_arg(list(v), subgraph_map, schema_type_hint=schema_type_hint)
    return _scalar_to_arg_value(v)


def _argument(
    v: object,
    subgraph_map: "dict[str, torch.fx.GraphModule] | None" = None,
    schema_type_hint: str | None = None,
) -> Argument:
    return Argument(
        value=_to_arg_value(v, subgraph_map, schema_type_hint=schema_type_hint)
    )


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
            # Required arg not provided; leave it out (invalid graph otherwise).
            continue
        # A written arg is one the op mutates in-place (schema Tensor(a!)); the op
        # schema is the source of truth (see comment on NamedArgument.mutated).
        mutated = sarg.alias_info is not None and sarg.alias_info.is_write
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
    out: dict[str, torch.fx.GraphModule] = {}
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            attr = getattr(graph_module, str(node.target), None)
            if isinstance(attr, torch.fx.GraphModule):
                out[node.name] = attr
    return out


_INPUT_KIND_MAP: dict[InputKind, SchemaInputKind] = {
    InputKind.PARAMETER: SchemaInputKind.PARAMETER,
    InputKind.BUFFER: SchemaInputKind.BUFFER,
    InputKind.CONSTANT_TENSOR: SchemaInputKind.CONSTANT_TENSOR,
}


def _is_non_persistent_buffer(ispec: object) -> bool:
    return getattr(ispec, "kind", None) == InputKind.BUFFER and not getattr(
        ispec, "persistent", True
    )


def _tensor_values_from(
    val_by_name: dict[str, torch.Tensor],
    names: list[str],
    exclude: set[str] | None = None,
) -> list[TensorValue]:
    seen: set[str] = set()
    out: list[TensorValue] = []
    for name in names:
        if name in seen:
            continue
        if exclude is not None and name in exclude:
            continue
        tensor = val_by_name.get(name)
        if tensor is None:
            continue
        seen.add(name)
        out.append(TensorValue(name=name, meta=_tensor_meta(tensor)))
    return out


def _is_getitem(fx_node: torch.fx.Node) -> bool:
    return fx_node.op == "call_function" and fx_node.target is operator.getitem


def _getitem_users(
    graph_module: torch.fx.GraphModule,
) -> dict[str, dict[int, torch.fx.Node]]:
    """Map each tuple-producing node name to {index: getitem_node} for its users."""
    users: dict[str, dict[int, torch.fx.Node]] = {}
    for fx_node in graph_module.graph.nodes:
        if not _is_getitem(fx_node):
            continue
        producer, idx = fx_node.args[0], fx_node.args[1]
        if isinstance(producer, torch.fx.Node) and isinstance(idx, int):
            users.setdefault(producer.name, {})[idx] = fx_node
    return users


def _is_tensor_list_return(ret_schema: Any) -> bool:
    rt = getattr(ret_schema, "real_type", None) or getattr(ret_schema, "type", None)
    return isinstance(rt, torch.ListType) and isinstance(
        rt.getElementType(), torch.TensorType
    )


def _scalar_output_kind(val: object) -> "OutputValueKind | None":
    if isinstance(val, (bool, torch.SymBool)):
        return OutputValueKind.BOOL
    if isinstance(val, (int, torch.SymInt)):
        return OutputValueKind.INT
    if isinstance(val, (float, torch.SymFloat)):
        return OutputValueKind.FLOAT
    return None


def _element_names(
    fx_node: torch.fx.Node,
    getitem_users: dict[str, dict[int, torch.fx.Node]],
    count: int,
) -> list[str]:
    """SSA names of a multi/list result's elements: the folded getitem user's name,
    or a synthetic name for a result no getitem extracts."""
    idx_map = getitem_users.get(fx_node.name, {})
    return [
        idx_map[i].name if i in idx_map else f"{fx_node.name}_out{i}"
        for i in range(count)
    ]


def _single_value_output(
    fx_node: torch.fx.Node, val: object, val_by_name: dict[str, torch.Tensor]
) -> Output:
    if isinstance(val, torch.Tensor):
        val_by_name[fx_node.name] = val
        alias_of = _output_alias_of(fx_node) if fx_node.op == "call_function" else None
        return Output(name=fx_node.name, alias_of=alias_of)
    kind = _scalar_output_kind(val)
    if kind is not None:
        return Output(name=fx_node.name, kind=kind)
    return Output(name=fx_node.name)


def _tuple_element_outputs(
    fx_node: torch.fx.Node,
    meta_val: object,
    getitem_users: dict[str, dict[int, torch.fx.Node]],
    val_by_name: dict[str, torch.Tensor],
) -> list[Output]:
    """One output per element of a tuple return (schema tuple or schema-less HOP).

    Nested tensor-list elements (a `Tensor[]` inside a tuple) are rejected; scalar
    elements carry an INT/BOOL/FLOAT kind.
    """
    idx_map = getitem_users.get(fx_node.name, {})
    outs: list[Output] = []
    for idx, m in enumerate(cast("list[object]", meta_val)):
        gi = idx_map.get(idx)
        name = gi.name if gi is not None else f"{fx_node.name}_out{idx}"
        if isinstance(m, (list, tuple)):
            raise ValueError(
                f"node {fx_node.name!r} return {idx} is a nested tensor-list; nested "
                f"list returns inside a tuple are not supported."
            )
        if isinstance(m, torch.Tensor):
            val_by_name[name] = m
            outs.append(Output(name=name))
            continue
        kind = _scalar_output_kind(m)
        if kind is None:
            raise ValueError(
                f"node {fx_node.name!r} return {idx} has unsupported type "
                f"{type(m).__name__}"
            )
        outs.append(Output(name=name, kind=kind))
    return outs


def _node_outputs(
    fx_node: torch.fx.Node,
    getitem_users: dict[str, dict[int, torch.fx.Node]],
    val_by_name: dict[str, torch.Tensor],
) -> list[Output]:
    """Outputs for a producer node, recording each result's tensor metadata.

    Schema-return driven (like torch._export.serde): a tuple of returns (e.g. topk,
    max.dim) becomes one output per return; a single ``Tensor[]`` return (e.g. split,
    unbind) becomes one TENSOR_LIST output over the element names; scalar returns
    carry an INT/BOOL/FLOAT kind. getitem users are folded in so their SSA names name
    the results. Schema-less tuple returns (HOPs like torch.cond) fold by meta['val'].
    """
    meta_val = fx_node.meta.get("val")
    op = _resolve_op_overload(fx_node.target) if fx_node.op == "call_function" else None
    if op is not None:
        returns = op._schema.returns
        if len(returns) == 0:
            return []
        if len(returns) == 1 and _is_tensor_list_return(returns[0]):
            elems = list(meta_val) if isinstance(meta_val, (list, tuple)) else []
            names = _element_names(fx_node, getitem_users, len(elems))
            for nm, m in zip(names, elems):
                if isinstance(m, torch.Tensor):
                    val_by_name[nm] = m
            return [Output(name="", kind=OutputValueKind.TENSOR_LIST, names=names)]
        if len(returns) > 1:
            return _tuple_element_outputs(fx_node, meta_val, getitem_users, val_by_name)
        return [_single_value_output(fx_node, meta_val, val_by_name)]
    if fx_node.op == "call_function" and isinstance(meta_val, (tuple, list)):
        return _tuple_element_outputs(fx_node, meta_val, getitem_users, val_by_name)
    return [_single_value_output(fx_node, meta_val, val_by_name)]


def _output_node_inputs(
    fx_node: torch.fx.Node,
    subgraph_map: dict[str, torch.fx.GraphModule],
) -> tuple[list[NamedArgument], list[str]]:
    out_vals = fx_node.args[0] if fx_node.args else ()
    out_list = list(out_vals) if isinstance(out_vals, (list, tuple)) else [out_vals]
    inputs: list[NamedArgument] = []
    output_names: list[str] = []
    for v in out_list:
        inputs.append(NamedArgument(name=None, arg=_argument(v, subgraph_map)))
        # This node's arguments are the full ordered result list (tensor refs, scalar
        # refs, and literals). Graph.outputs lists only the tensor values, which carry
        # metadata and can be mutation targets.
        if isinstance(v, torch.fx.Node) and isinstance(v.meta.get("val"), torch.Tensor):
            output_names.append(v.name)
    return inputs, output_names


def _collect_fx_nodes(
    graph_module: torch.fx.GraphModule,
    subgraph_map: dict[str, torch.fx.GraphModule],
) -> tuple[list[Node], dict[str, torch.Tensor], list[str]]:
    nodes: list[Node] = []
    val_by_name: dict[str, torch.Tensor] = {}
    output_names: list[str] = []
    getitem_users = _getitem_users(graph_module)

    for fx_node in graph_module.graph.nodes:
        if fx_node.op == "get_attr":
            if fx_node.name in subgraph_map:
                continue
            raise ValueError(
                f"get_attr node {fx_node.name!r} (target {fx_node.target!r}) cannot be "
                f"serialized: the graph is not lifted. Serialize a lifted "
                f"torch.export ExportedProgram (params/buffers/constants as inputs) "
                f"instead of an unlifted module."
            )

        # getitem users of a tuple-producing node are folded into that node's
        # outputs (see _node_outputs), so they are not emitted as their own nodes.
        if _is_getitem(fx_node):
            producer = fx_node.args[0]
            if isinstance(producer, torch.fx.Node) and isinstance(
                producer.meta.get("val"), (tuple, list)
            ):
                continue

        kind = _op_kind(fx_node.op)
        target = None
        inputs: list[NamedArgument] = []
        outputs: list[Output] = []

        if fx_node.op == "output":
            inputs, node_output_names = _output_node_inputs(fx_node, subgraph_map)
            output_names.extend(node_output_names)
        else:
            if fx_node.op == "call_function":
                target = serialize_operator(fx_node.target)
                inputs = _named_arguments(fx_node, subgraph_map)
            outputs = _node_outputs(fx_node, getitem_users, val_by_name)

        nodes.append(
            Node(
                name=fx_node.name,
                op_kind=kind,
                target=target,
                inputs=inputs or None,
                outputs=outputs or None,
            )
        )

    return nodes, val_by_name, output_names


def _validate_tensor_user_inputs(graph_signature: object) -> None:
    for ispec in getattr(graph_signature, "input_specs", []) or []:
        if ispec.kind == InputKind.USER_INPUT and not isinstance(
            ispec.arg, TensorArgument
        ):
            raise ValueError(
                f"unsupported non-tensor user input {ispec.arg!r}; "
                f"only tensor user inputs are supported."
            )


def _build_output_specs(
    output_names: list[str],
    graph_signature: object,
) -> tuple[list[OutputSpec], set[str]]:
    params_to_mutate = getattr(graph_signature, "parameters_to_mutate", None) or {}
    if params_to_mutate:
        raise ValueError(
            f"parameter mutation is not supported: {sorted(params_to_mutate)}. "
            f"Serialize a graph that does not mutate parameters."
        )
    buffers_to_mutate = getattr(graph_signature, "buffers_to_mutate", None) or {}
    user_inputs_to_mutate = (
        getattr(graph_signature, "user_inputs_to_mutate", None) or {}
    )
    specs: list[OutputSpec] = []
    for name in output_names:
        if name in buffers_to_mutate:
            specs.append(
                OutputSpec(
                    name=name,
                    kind=OutputKind.BUFFER_MUTATION,
                    target=buffers_to_mutate[name],
                )
            )
        elif name in user_inputs_to_mutate:
            specs.append(
                OutputSpec(
                    name=name,
                    kind=OutputKind.USER_INPUT_MUTATION,
                    target=user_inputs_to_mutate[name],
                )
            )
        else:
            specs.append(OutputSpec(name=name, kind=OutputKind.USER_OUTPUT))
    mutated_fqns = set(buffers_to_mutate.values())
    return specs, mutated_fqns


def _extract_constants_and_mutable_buffers(
    graph_signature: object,
    state_dict: dict[str, object],
    constants: dict[str, object] | None,
    mutated_fqns: set[str],
) -> tuple[list[NamedTensorRef], dict[str, torch.Tensor], list[MutableBufferSpec]]:
    constant_refs: list[NamedTensorRef] = []
    constant_data: dict[str, torch.Tensor] = {}
    mutable_buffers: list[MutableBufferSpec] = []

    for ispec in getattr(graph_signature, "input_specs", []) or []:
        if ispec.kind not in _INPUT_KIND_MAP:
            continue
        name = getattr(getattr(ispec, "arg", None), "name", None)
        target_fqn = getattr(ispec, "target", None)
        if name is None or target_fqn is None:
            continue
        if _is_non_persistent_buffer(ispec):
            mutable_buffers.append(MutableBufferSpec(name=name, fqn=target_fqn))
            continue
        tensor = None
        if target_fqn in state_dict:
            tensor = state_dict[target_fqn]
        elif constants is not None and target_fqn in constants:
            tensor = constants[target_fqn]
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor = tensor.contiguous()
        constant_refs.append(
            NamedTensorRef(
                name=name,
                data_key=target_fqn,
                meta=_tensor_meta(tensor),
                kind=_INPUT_KIND_MAP[ispec.kind],
                mutated=target_fqn in mutated_fqns,
            )
        )
        constant_data[target_fqn] = tensor

    return constant_refs, constant_data, mutable_buffers


def _build_subgraph(
    graph_module: torch.fx.GraphModule,
    nodes: list[Node],
    val_by_name: dict[str, torch.Tensor],
    output_names: list[str],
) -> tuple[
    Graph,
    list[NamedTensorRef],
    list[OutputSpec],
    list[MutableBufferSpec],
    dict[str, torch.Tensor],
]:
    user_inputs = [n.name for n in graph_module.graph.nodes if n.op == "placeholder"]
    tensor_values = _tensor_values_from(val_by_name, list(val_by_name.keys()))
    graph = Graph(
        nodes=nodes,
        inputs=user_inputs or None,
        outputs=output_names or None,
        tensor_values=tensor_values or None,
    )
    return graph, [], [], [], {}


def _build_method_graph(
    graph_module: torch.fx.GraphModule,
    nodes: list[Node],
    val_by_name: dict[str, torch.Tensor],
    output_names: list[str],
    graph_signature: object,
    state_dict: dict[str, object],
    constants: dict[str, object] | None,
) -> tuple[
    Graph,
    list[NamedTensorRef],
    list[OutputSpec],
    list[MutableBufferSpec],
    dict[str, torch.Tensor],
]:
    _validate_tensor_user_inputs(graph_signature)
    output_specs, mutated_fqns = _build_output_specs(output_names, graph_signature)
    constant_refs, constant_data, mutable_buffers = (
        _extract_constants_and_mutable_buffers(
            graph_signature, state_dict, constants, mutated_fqns
        )
    )
    constant_names = {c.name for c in constant_refs}
    tensor_values = _tensor_values_from(
        val_by_name,
        [n for n in val_by_name if n not in constant_names],
    )
    user_inputs = list(getattr(graph_signature, "user_inputs", []) or [])
    graph = Graph(
        nodes=nodes,
        inputs=user_inputs or None,
        outputs=output_names or None,
        tensor_values=tensor_values or None,
    )
    return graph, constant_refs, output_specs, mutable_buffers, constant_data


def _build_graph_body(
    graph_module: torch.fx.GraphModule,
    graph_signature: object | None,
    state_dict: dict[str, object],
    constants: dict[str, object] | None,
) -> tuple[
    Graph,
    list[NamedTensorRef],
    list[OutputSpec],
    list[MutableBufferSpec],
    dict[str, torch.Tensor],
]:
    subgraph_map = _subgraph_map(graph_module)
    nodes, val_by_name, output_names = _collect_fx_nodes(graph_module, subgraph_map)

    if graph_signature is None:
        return _build_subgraph(graph_module, nodes, val_by_name, output_names)

    return _build_method_graph(
        graph_module,
        nodes,
        val_by_name,
        output_names,
        graph_signature,
        state_dict,
        constants,
    )


# ---------------------------------------------------------------------------
# JSON to/from flatbuffer via flatc.
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


_FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float16,
    torch.float64,
    torch.bfloat16,
)


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return (
            statically_known_true(a.storage_offset() == b.storage_offset())
            and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        )
    except Exception:
        return False


def _float_equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        if not (torch.isnan(a).any() or torch.isnan(b).any()):
            return False
    except RuntimeError:
        return False
    eq = (a == b) | (torch.isnan(a) & torch.isnan(b))
    return bool(eq.all().item())


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a is b:
        return True
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if _same_storage(a, b):
        return True

    ca = a.detach()
    cb = b.detach()

    if ca.device != cb.device:
        if ca.device.type != "cpu":
            ca = ca.cpu()
        if cb.device.type != "cpu":
            cb = cb.cpu()

    if torch.equal(ca, cb):
        return True

    if ca.dtype in _FLOAT_DTYPES:
        return _float_equal_with_nan(ca, cb)
    return False


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
    caller ships constant_data as the external constant file.

    Cross-method sharing is by fqn: bundling methods asserts they come from a single
    model namespace, so an fqn is the same buffer/constant everywhere. That assertion
    is validated; if two methods carry the same fqn with different data (constants) or
    different shape/dtype (mutable buffers), this raises rather than silently aliasing
    or clobbering.
    """
    method_objs: list[Method] = []
    constant_data: dict[str, torch.Tensor] = {}
    mutable_meta: dict[str, TensorMeta] = {}
    for name, (graph_module, graph_signature, state_dict, constants) in methods.items():
        graph, constant_refs, output_specs, mutable_buffers, cdata = _build_graph_body(
            graph_module, graph_signature, state_dict, constants
        )
        method_objs.append(
            Method(
                name=name,
                graph=graph,
                constants=constant_refs or None,
                output_specs=output_specs or None,
                mutable_buffers=mutable_buffers or None,
            )
        )

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
        for mb in mutable_buffers:
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


def _check_optional_tensor_list_refs(
    value: OptionalTensorListArg, ctx: str, check_ref: Any
) -> None:
    if len(value.names) != len(value.has_value):
        raise ValueError(
            f"{ctx}: OptionalTensorListArg names length {len(value.names)} does not "
            f"match has_value length {len(value.has_value)}"
        )
    for n, present in zip(value.names, value.has_value):
        if present:
            check_ref(n, ctx)


def _check_int_list_refs(value: IntListArg, ctx: str, check_ref: Any) -> None:
    refs = value.refs
    # refs is parallel to values; a set refs[i] references an in-graph int.
    if refs is None:
        return
    if len(refs) != len(value.values):
        raise ValueError(
            f"{ctx}: IntListArg refs length {len(refs)} does not "
            f"match values length {len(value.values)}"
        )
    for r in refs:
        if r:
            check_ref(r, ctx)


def _check_arg_refs(value: ArgumentValue, ctx: str, check_ref: Any) -> None:
    if isinstance(value, TensorArg):
        check_ref(value.name, ctx)
    elif isinstance(value, TensorListArg):
        for n in value.names:
            check_ref(n, ctx)
    elif isinstance(value, OptionalTensorListArg):
        _check_optional_tensor_list_refs(value, ctx, check_ref)
    elif isinstance(value, IntArg):
        # ref, when set, is an SSA reference to an in-graph scalar value.
        if value.ref:
            check_ref(value.ref, ctx)
    elif isinstance(value, (BoolArg, FloatArg)):
        if value.ref:
            check_ref(value.ref, ctx)
    elif isinstance(value, IntListArg):
        _check_int_list_refs(value, ctx, check_ref)
    elif isinstance(value, GraphArg):
        # HOP subgraph: self-contained (its own placeholder namespace).
        validate_graph(value.graph)


def _check_io_meta(graph: Graph, meta_names: set[str]) -> None:
    for name in graph.inputs or []:
        if name not in meta_names:
            raise ValueError(f"input {name!r} missing tensor metadata")
    for name in graph.outputs or []:
        if name not in meta_names:
            raise ValueError(f"output {name!r} missing tensor metadata")


def _define_node_outputs(node: Node, define: Any) -> None:
    ctx = f"node {node.name!r}"
    for out in node.outputs or []:
        if out.kind == OutputValueKind.TENSOR_LIST:
            for nm in out.names or []:
                define(nm, ctx)
        else:
            define(out.name, ctx)


def _validate_node(node: Node, defined: set[str], define: Any, check_ref: Any) -> None:
    ctx = f"node {node.name!r}"
    for na in node.inputs or []:
        _check_arg_refs(na.arg.value, ctx, check_ref)
    for out in node.outputs or []:
        if out.alias_of:
            check_ref(out.alias_of, f"{ctx} output alias")
    # A placeholder must restate a value already declared as an input or external
    # binding; any other node defines a fresh value per output (a multi-output op
    # defines one value per result).
    if node.op_kind == OpKind.PLACEHOLDER:
        if node.name not in defined:
            raise ValueError(
                f"{ctx}: placeholder is not a declared input or external binding"
            )
    else:
        _define_node_outputs(node, define)


def validate_graph(graph: Graph, defined_extra: set[str] | None = None) -> None:
    """Assert a (pure) graph body is structurally self-contained: every value
    reference resolves and every input/output has tensor metadata. Recurses into
    every ``GraphArg`` subgraph. Raises ``ValueError`` on the first inconsistency.

    defined_extra supplies names defined outside the body (a method's constants and
    mutable buffers, see validate_method); a HOP subgraph passes None since its
    operands are its own placeholders. Method-level bindings (constant data
    availability, mutable-buffer metadata) are checked in validate_method.
    """
    meta_names = {tv.name for tv in (graph.tensor_values or [])}
    defined: set[str] = set()
    seen_nodes: set[str] = set()

    def define(name: str, ctx: str) -> None:
        if name in defined:
            raise ValueError(f"{ctx}: duplicate value definition {name!r}")
        defined.add(name)

    for name in graph.inputs or []:
        define(name, "input")
    for name in sorted(defined_extra or ()):
        define(name, "external binding")

    def check_ref(name: str, ctx: str) -> None:
        if name and name not in defined:
            raise ValueError(f"{ctx}: unresolved value reference {name!r}")

    # Define a node's values only after its inputs resolve, so forward references and
    # cycles are rejected (nodes must be topologically ordered). Node names are unique
    # across the graph.
    for node in graph.nodes:
        if node.name in seen_nodes:
            raise ValueError(f"node {node.name!r}: duplicate node name")
        seen_nodes.add(node.name)
        _validate_node(node, defined, define, check_ref)

    for name in graph.outputs or []:
        check_ref(name, "output")

    _check_io_meta(graph, meta_names)


def validate_method(
    method: Method, available_data_keys: set[str] | None = None
) -> None:
    """Validate a method: its graph body (see ``validate_graph``) plus its
    constant/mutable-buffer bindings.

    This enforces that the method graph plus its external constant data (whose keys
    are available_data_keys) carry everything needed to run.
    Names bound by constants / mutable buffers are supplied to the body's reference
    check; every mutable buffer must have tensor metadata; and, when
    ``available_data_keys`` is given, every ``NamedTensorRef.data_key`` must be
    present. Mutable buffers are not data-backed, so they are exempt from the
    data-keys check.
    """
    defined_extra: set[str] = {c.name for c in (method.constants or [])}
    defined_extra.update(mb.name for mb in (method.mutable_buffers or []))
    validate_graph(method.graph, defined_extra=defined_extra)

    meta_names = {tv.name for tv in (method.graph.tensor_values or [])}
    for mb in method.mutable_buffers or []:
        if mb.name not in meta_names:
            raise ValueError(
                f"mutable buffer {mb.name!r} (fqn {mb.fqn!r}) missing tensor metadata"
            )

    if available_data_keys is not None:
        for c in method.constants or []:
            if c.data_key not in available_data_keys:
                raise ValueError(
                    f"constant {c.name!r} (data_key {c.data_key!r}) has no data in the "
                    f"provided external constant keys"
                )


def validate_program(
    program: Program, available_data_keys: set[str] | None = None
) -> None:
    """Validate every method (see ``validate_method``)."""
    for method in program.methods:
        validate_method(method, available_data_keys)
