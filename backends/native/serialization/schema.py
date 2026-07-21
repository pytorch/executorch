# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Python dataclass mirror of native_graph.fbs.

These classes are the AOT representation of the generic fx-graph flatbuffer.
They are serialized to JSON via ``executorch.exir._serialize._dataclass._DataclassEncoder``
and compiled to a flatbuffer with ``flatc`` (see graph_serialize.py).

Convention (matches executorch/exir/schema.py):
  - Union-typed fields MUST be annotated as string literals (e.g. ``"ArgumentValue"``)
    so ``_DataclassEncoder`` emits the ``<field>_type`` discriminator that flatc's
    JSON union format requires. The union member dataclass names MUST equal the
    corresponding flatbuffer table names.
  - All other fields use real type annotations.
  - Fields that are ``required`` in the .fbs are non-optional here; everything else
    is Optional with a default so the flatc --json round-trip (which omits unset
    vectors) deserializes cleanly.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union


class ScalarType(IntEnum):
    BYTE = 0
    CHAR = 1
    SHORT = 2
    INT = 3
    LONG = 4
    HALF = 5
    FLOAT = 6
    DOUBLE = 7
    BOOL = 11
    BFLOAT16 = 15
    UINT16 = 16
    UINT32 = 17
    UINT64 = 18


class OpKind(IntEnum):
    CALL_FUNCTION = 0
    PLACEHOLDER = 1
    OUTPUT = 2


class InputKind(IntEnum):
    USER_INPUT = 0
    PARAMETER = 1
    BUFFER = 2
    CONSTANT_TENSOR = 3


class OutputKind(IntEnum):
    USER_OUTPUT = 0
    BUFFER_MUTATION = 1
    USER_INPUT_MUTATION = 2


@dataclass
class SymInt:
    as_int: int = 0
    as_symbol: Optional[str] = None


@dataclass
class TensorMeta:
    dtype: ScalarType
    sizes: List[SymInt]
    strides: List[SymInt]


@dataclass
class TensorValue:
    name: str
    meta: TensorMeta


# ---------------------------------------------------------------------------
# Argument union members. Class names must match the .fbs table names.
# ---------------------------------------------------------------------------


@dataclass
class TensorArg:
    name: str


@dataclass
class NoneArg:
    pass


@dataclass
class IntArg:
    value: int


@dataclass
class FloatArg:
    value: float


@dataclass
class BoolArg:
    value: bool


@dataclass
class StringArg:
    value: str


@dataclass
class ScalarTypeArg:
    value: ScalarType


@dataclass
class SymIntArg:
    value: SymInt


@dataclass
class IntListArg:
    values: List[int]


@dataclass
class SymIntListArg:
    values: List[SymInt]


@dataclass
class FloatListArg:
    values: List[float]


@dataclass
class BoolListArg:
    values: List[bool]


@dataclass
class TensorListArg:
    names: List[str]


@dataclass
class OptionalTensorListArg:
    names: List[str]
    has_value: List[bool]


@dataclass
class Argument:
    # Union types must be specified as strings so _DataclassEncoder can see them.
    # ArgumentValue is defined below (after Graph) because one of its members,
    # GraphArg, holds a nested Graph (recursive schema).
    value: "ArgumentValue"


@dataclass
class NamedArgument:
    arg: Argument
    name: Optional[str] = None
    # True if the op writes this input in-place (schema Tensor(a!)).
    mutated: bool = False


@dataclass
class KeyValue:
    key: str
    value: str


# A value produced by a node. `alias_of`, when set, is the SSA name of an input
# value this output shares storage with (op-schema view annotation).
@dataclass
class Output:
    name: str
    alias_of: Optional[str] = None


@dataclass
class Node:
    name: str
    op_kind: OpKind
    target: Optional[str] = None
    inputs: Optional[List[NamedArgument]] = None
    outputs: Optional[List[Output]] = None
    metadata: Optional[List[KeyValue]] = None
    debug_handle: Optional[int] = None


@dataclass
class ConstantRef:
    name: str
    fqn: str
    meta: TensorMeta
    kind: InputKind = InputKind.CONSTANT_TENSOR
    mutated: bool = False


@dataclass
class OutputSpec:
    name: str
    kind: OutputKind = OutputKind.USER_OUTPUT
    target: Optional[str] = None


# A non-data-backed mutable buffer (e.g. a zero-initialized KV cache): graph state
# that is neither a user input nor a ConstantRef. Shape/dtype live in
# tensor_values (keyed by `name`); cross-method sharing is by `fqn`.
@dataclass
class MutableBufferSpec:
    name: str
    fqn: str


@dataclass
class Graph:
    nodes: List[Node]
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    tensor_values: Optional[List[TensorValue]] = field(default=None)
    constants: Optional[List[ConstantRef]] = field(default=None)
    output_specs: Optional[List[OutputSpec]] = field(default=None)
    mutable_buffers: Optional[List[MutableBufferSpec]] = field(default=None)


# A subgraph passed to a higher-order op (torch.cond / while_loop / map). Inlined
# nested graph (mirrors torch._export.serde's GraphArgument); makes Graph
# recursive. Defined after Graph so the `graph` annotation is a real class
# reference (a string forward-ref would not deserialize as a nested dataclass).
@dataclass
class GraphArg:
    name: str
    graph: Graph


# BC: APPEND ONLY — keep in sync with the union in native_graph.fbs.
ArgumentValue = Union[
    TensorArg,
    NoneArg,
    IntArg,
    FloatArg,
    BoolArg,
    StringArg,
    ScalarTypeArg,
    SymIntArg,
    IntListArg,
    FloatListArg,
    BoolListArg,
    TensorListArg,
    OptionalTensorListArg,
    SymIntListArg,
    GraphArg,
]


@dataclass
class Method:
    name: str
    graph: Graph


@dataclass
class Program:
    methods: List[Method]
    version: Optional[str] = None
