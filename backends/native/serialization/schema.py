# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Python dataclass mirror of native_graph.fbs, the source of truth for semantics.

Serialized to JSON by executorch.exir._serialize._dataclass._DataclassEncoder and
compiled to a flatbuffer with flatc (see graph_serialize.py).

Conventions (match executorch/exir/schema.py):
  - Union fields are annotated as string literals (e.g. "ArgumentValue") so the
    encoder emits the <field>_type discriminator that flatc's JSON unions need.
    Union member class names must equal the flatbuffer table names.
  - required fbs fields are non-optional here; the rest are Optional with a default
    so the flatc --json round-trip (which omits unset vectors) deserializes.

See native_graph.fbs for field semantics and name scoping.
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


class OutputValueKind(IntEnum):
    # Discriminates what a Node.Output produces. TENSOR (default) keeps the legacy
    # single-tensor shape (name + optional alias_of). TENSOR_LIST is a `Tensor[]`
    # return whose element SSA names are in `names` (name empty). INT/BOOL/FLOAT are
    # scalar returns whose SSA name is `name`.
    TENSOR = 0
    TENSOR_LIST = 1
    INT = 2
    BOOL = 3
    FLOAT = 4


# Quantization. A tensor's quant scheme rides on its TensorMeta, so it applies to
# graph I/O, intermediates, and constants uniformly. Absent means not quantized;
# dtype stays the storage dtype and the scheme says how to interpret it. The scheme
# set is append-only and grows over time.


@dataclass
class AffineGroup:
    # Affine group-wise quant along the last axis. See native_graph.fbs for the full
    # dequant and storage contract (unsigned qdata, quant_min offset, out-of-line
    # scales, packed byte length).
    scale_data_key: str
    scale_dtype: ScalarType
    quant_min: int
    quant_max: int
    group_size: int = 0
    zero_point_data_key: Optional[str] = None
    zero_point_dtype: ScalarType = ScalarType.INT


@dataclass
class PackedQuant:
    # Opaque codec-defined packed layout (e.g. "gguf:q4k", "mxfp4", "nvfp4"). Block
    # size, bit width, and scales are all implied by `codec`. See native_graph.fbs.
    codec: str


# Append-only; keep in sync with the union in native_graph.fbs.
QuantScheme = Union[
    AffineGroup,
    PackedQuant,
]


@dataclass
class QuantSpec:
    # Wrapper so the scheme union can be optional on TensorMeta: the JSON codec only
    # supports required string-annotated unions, so an optional union is nested in
    # this table (cf. EValue/KernelTypes in exir).
    scheme: "QuantScheme"


@dataclass
class Dim:
    # A tensor dimension range. Static: min == max. Dynamic: min < max, or max < 0
    # for unbounded. No symbols; the runtime uses concrete shapes and plans from max.
    min: int
    max: int = -1


@dataclass
class TensorMeta:
    # sizes is the logical shape, dtype the element type. When quant is set the tensor
    # is packed (dtype BYTE) and the physical layout and byte length come from the
    # quant scheme and its external constant blob, not prod(sizes).
    dtype: ScalarType
    sizes: List[Dim]
    # Memory layout as a permutation of dim indices, outermost first (contiguous is
    # [0, 1, ..., n-1]). Independent of the sizes. Empty means contiguous.
    dim_order: Optional[List[int]] = None
    quant: Optional[QuantSpec] = None


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
    # Non-empty means a reference to an in-graph int value by SSA name (a sym_size or
    # arith node), and value is ignored. Empty means the literal value.
    ref: Optional[str] = None


@dataclass
class FloatArg:
    value: float
    # Non-empty ref references an in-graph float value by SSA name; value is ignored.
    ref: Optional[str] = None


@dataclass
class BoolArg:
    value: bool
    # Non-empty ref references an in-graph bool value by SSA name; value is ignored.
    ref: Optional[str] = None


@dataclass
class StringArg:
    value: str


@dataclass
class ScalarTypeArg:
    value: ScalarType


@dataclass
class IntListArg:
    values: List[int]
    # Parallel to values: a non-empty refs[i] makes element i a reference to an
    # in-graph int value by SSA name (values[i] ignored). Empty means all literal.
    refs: Optional[List[str]] = None


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
    # String annotation so the encoder emits the union discriminator. ArgumentValue is
    # defined after Graph because GraphArg holds a nested Graph (recursive schema).
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


# `alias_of`, when set, is the SSA name of an input value this output shares storage
# with (op-schema view annotation). `kind` discriminates the produced value: a
# TENSOR_LIST (`Tensor[]` return) carries its element SSA names in `names` with an
# empty `name`; INT/BOOL/FLOAT are scalar returns named by `name`.
@dataclass
class Output:
    name: str
    alias_of: Optional[str] = None
    kind: OutputValueKind = OutputValueKind.TENSOR
    names: Optional[List[str]] = None


@dataclass
class Node:
    name: str
    op_kind: OpKind
    target: Optional[str] = None
    inputs: Optional[List[NamedArgument]] = None
    outputs: Optional[List[Output]] = None
    metadata: Optional[List[KeyValue]] = None


@dataclass
class NamedTensorRef:
    name: str
    data_key: str
    meta: TensorMeta
    kind: InputKind = InputKind.CONSTANT_TENSOR
    mutated: bool = False


@dataclass
class OutputSpec:
    name: str
    kind: OutputKind = OutputKind.USER_OUTPUT
    target: Optional[str] = None


# Mutable buffer that is graph state but not data-backed (e.g. a zero-initialized KV
# cache). Shape and dtype live in tensor_values; cross-method sharing is by fqn.
@dataclass
class MutableBufferSpec:
    name: str
    fqn: str


@dataclass
class Graph:
    # Pure function body, used for both a top-level method and a HOP subgraph.
    # Method-level bindings (constants, output_specs, mutable_buffers) live on Method.
    nodes: List[Node]
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    tensor_values: Optional[List[TensorValue]] = field(default=None)


# Subgraph passed to a higher-order op (torch.cond / while_loop / map), inlined as a
# nested Graph. Defined after Graph so `graph` is a real class reference; a string
# forward-ref would not deserialize as a nested dataclass.
@dataclass
class GraphArg:
    name: str
    graph: Graph


# Append-only; keep in sync with the union in native_graph.fbs.
ArgumentValue = Union[
    TensorArg,
    NoneArg,
    IntArg,
    FloatArg,
    BoolArg,
    StringArg,
    ScalarTypeArg,
    IntListArg,
    FloatListArg,
    BoolListArg,
    TensorListArg,
    OptionalTensorListArg,
    GraphArg,
]


@dataclass
class Method:
    name: str
    graph: Graph
    # Constant/parameter/buffer references binding a method-graph placeholder to a
    # data_key. A HOP subgraph has none; its params are lifted here as operands.
    constants: Optional[List[NamedTensorRef]] = None
    # Per-output classification (user output vs buffer/user-input mutation), parallel
    # to graph.outputs by name.
    output_specs: Optional[List[OutputSpec]] = None
    # Non-persistent mutable buffers (e.g. KV caches): no shipped data, zero-init.
    mutable_buffers: Optional[List[MutableBufferSpec]] = None


@dataclass
class Program:
    methods: List[Method]
    version: Optional[str] = None
