# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Diff two ExecuTorch .pte files and report structural/data differences."""

import argparse
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import (
    Bool,
    BoolList,
    DelegateCall,
    Double,
    DoubleList,
    ExecutionPlan,
    ExtraTensorInfo,
    Int,
    IntList,
    KernelCall,
    Null,
    OptionalTensorList,
    String,
    Tensor,
    TensorList,
)

# ScalarType -> (struct format char, element size in bytes)
_SCALAR_TYPE_TO_FORMAT: Dict[ScalarType, Tuple[str, int]] = {
    ScalarType.BYTE: ("B", 1),
    ScalarType.CHAR: ("b", 1),
    ScalarType.SHORT: ("h", 2),
    ScalarType.INT: ("i", 4),
    ScalarType.LONG: ("q", 8),
    ScalarType.HALF: ("e", 2),
    ScalarType.FLOAT: ("f", 4),
    ScalarType.DOUBLE: ("d", 8),
    ScalarType.BOOL: ("?", 1),
    ScalarType.QUINT8: ("B", 1),
    ScalarType.QINT8: ("b", 1),
    ScalarType.QINT32: ("i", 4),
}

_SCALAR_TYPE_NAMES: Dict[ScalarType, str] = {
    ScalarType.BYTE: "byte",
    ScalarType.CHAR: "char",
    ScalarType.SHORT: "short",
    ScalarType.INT: "int",
    ScalarType.LONG: "long",
    ScalarType.HALF: "half",
    ScalarType.FLOAT: "float",
    ScalarType.DOUBLE: "double",
    ScalarType.COMPLEX32: "complex32",
    ScalarType.COMPLEX64: "complex64",
    ScalarType.COMPLEX128: "complex128",
    ScalarType.BOOL: "bool",
    ScalarType.QINT8: "qint8",
    ScalarType.QUINT8: "quint8",
    ScalarType.QINT32: "qint32",
    ScalarType.BFLOAT16: "bfloat16",
    ScalarType.QUINT4x2: "quint4x2",
    ScalarType.QUINT2x4: "quint2x4",
}

_EVALUE_TYPE_NAMES = {
    Null: "Null",
    Int: "Int",
    Bool: "Bool",
    Double: "Double",
    String: "String",
    Tensor: "Tensor",
    IntList: "IntList",
    DoubleList: "DoubleList",
    BoolList: "BoolList",
    TensorList: "TensorList",
    OptionalTensorList: "OptionalTensorList",
}


@dataclass
class OperatorUsage:
    name: str
    arg_index: int


@dataclass
class TensorInfo:
    evalue_index: int
    scalar_type: ScalarType
    sizes: List[int]
    fqn: Optional[str]
    data_buffer_idx: int
    operator_usages: List[OperatorUsage] = field(default_factory=list)


@dataclass
class FieldDiff:
    field_name: str
    value_a: Any
    value_b: Any


@dataclass
class ElementDiff:
    flat_index: int
    multi_index: Tuple[int, ...]
    value_a: float
    value_b: float


@dataclass
class TensorDataDiff:
    tensor_a: TensorInfo
    tensor_b: TensorInfo
    byte_size_a: int
    byte_size_b: int
    bytes_differ: bool
    metadata_diffs: List[FieldDiff] = field(default_factory=list)
    num_elements: int = 0
    num_differing: int = 0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    element_diffs: List[ElementDiff] = field(default_factory=list)


@dataclass
class EValueDiff:
    evalue_index: int
    type_a: str
    type_b: str
    type_mismatch: bool = False
    field_diffs: List[FieldDiff] = field(default_factory=list)
    operator_usages_a: List[OperatorUsage] = field(default_factory=list)
    operator_usages_b: List[OperatorUsage] = field(default_factory=list)


@dataclass
class ExecutionPlanDiff:
    plan_index: int
    name_a: Optional[str] = None
    name_b: Optional[str] = None
    operators_only_in_a: List[str] = field(default_factory=list)
    operators_only_in_b: List[str] = field(default_factory=list)
    delegates_only_in_a: List[str] = field(default_factory=list)
    delegates_only_in_b: List[str] = field(default_factory=list)
    non_const_buffer_sizes_a: Optional[List[int]] = None
    non_const_buffer_sizes_b: Optional[List[int]] = None
    value_count_a: Optional[int] = None
    value_count_b: Optional[int] = None
    instruction_count_a: Optional[int] = None
    instruction_count_b: Optional[int] = None
    tensor_diffs: List[TensorDataDiff] = field(default_factory=list)
    evalue_diffs: List[EValueDiff] = field(default_factory=list)
    tensors_only_in_a: List[TensorInfo] = field(default_factory=list)
    tensors_only_in_b: List[TensorInfo] = field(default_factory=list)

    def has_differences(self) -> bool:
        return bool(
            self.name_a is not None
            or self.operators_only_in_a
            or self.operators_only_in_b
            or self.delegates_only_in_a
            or self.delegates_only_in_b
            or self.non_const_buffer_sizes_a is not None
            or self.value_count_a is not None
            or self.instruction_count_a is not None
            or self.tensor_diffs
            or self.evalue_diffs
            or self.tensors_only_in_a
            or self.tensors_only_in_b
        )


@dataclass
class NamedDataDiff:
    key: str
    only_in: str  # "A" or "B"
    bytes_differ: bool = False


@dataclass
class PTEDiffResult:
    path_a: str
    path_b: str
    size_a: int
    size_b: int
    bitwise_equal: bool
    version_a: Optional[int] = None
    version_b: Optional[int] = None
    plan_diffs: List[ExecutionPlanDiff] = field(default_factory=list)
    extra_plans_in_a: List[str] = field(default_factory=list)
    extra_plans_in_b: List[str] = field(default_factory=list)
    named_data_diffs: List[NamedDataDiff] = field(default_factory=list)
    error: Optional[str] = None


def _build_tensor_to_operators(
    plan: ExecutionPlan,
) -> Dict[int, List[OperatorUsage]]:
    mapping: Dict[int, List[OperatorUsage]] = {}
    operators = plan.operators
    for chain in plan.chains:
        for instr in chain.instructions:
            args = instr.instr_args
            if isinstance(args, KernelCall):
                op = operators[args.op_index]
                opname = f"{op.name}.{op.overload}" if op.overload else op.name
                for arg_idx, evalue_idx in enumerate(args.args):
                    mapping.setdefault(evalue_idx, []).append(
                        OperatorUsage(name=opname, arg_index=arg_idx)
                    )
            elif isinstance(args, DelegateCall):
                delegate = plan.delegates[args.delegate_index]
                for arg_idx, evalue_idx in enumerate(args.args):
                    mapping.setdefault(evalue_idx, []).append(
                        OperatorUsage(name=f"delegate:{delegate.id}", arg_index=arg_idx)
                    )
    return mapping


def _get_tensor_bytes(
    tensor: Tensor,
    constant_buffer: list,
    mutable_data: Optional[list],
) -> Optional[bytes]:
    if tensor.data_buffer_idx > 0 and tensor.data_buffer_idx < len(constant_buffer):
        return constant_buffer[tensor.data_buffer_idx].storage
    if (
        tensor.extra_tensor_info
        and tensor.extra_tensor_info.mutable_data_segments_idx > 0
        and mutable_data is not None
        and tensor.extra_tensor_info.mutable_data_segments_idx < len(mutable_data)
    ):
        return mutable_data[tensor.extra_tensor_info.mutable_data_segments_idx].storage
    return None


def _unravel_index(flat_idx: int, sizes: List[int]) -> Tuple[int, ...]:
    if not sizes:
        return ()
    indices = []
    for dim in reversed(sizes):
        indices.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(indices))


def _compute_tensor_stats(
    bytes_a: bytes,
    bytes_b: bytes,
    scalar_type: ScalarType,
    sizes: List[int],
    max_samples: int = 10,
) -> Tuple[int, int, float, float, List[ElementDiff]]:
    """Returns (num_elements, num_differing, max_abs_diff, mean_abs_diff, element_diffs)."""
    fmt_info = _SCALAR_TYPE_TO_FORMAT.get(scalar_type)
    if fmt_info is None:
        num_bytes = min(len(bytes_a), len(bytes_b))
        num_diff = sum(1 for i in range(num_bytes) if bytes_a[i] != bytes_b[i])
        return (num_bytes, num_diff, 0.0, 0.0, [])

    fmt_char, elem_size = fmt_info
    num_elements = 1
    for s in sizes:
        num_elements *= s
    actual_count_a = len(bytes_a) // elem_size
    actual_count_b = len(bytes_b) // elem_size
    count = min(num_elements, actual_count_a, actual_count_b)

    num_differing = 0
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    element_diffs: List[ElementDiff] = []

    for i in range(count):
        offset = i * elem_size
        (val_a,) = struct.unpack_from(f"<{fmt_char}", bytes_a, offset)
        (val_b,) = struct.unpack_from(f"<{fmt_char}", bytes_b, offset)
        if val_a != val_b:
            diff = abs(float(val_a) - float(val_b))
            num_differing += 1
            max_abs_diff = max(max_abs_diff, diff)
            total_abs_diff += diff
            if len(element_diffs) < max_samples:
                element_diffs.append(
                    ElementDiff(
                        flat_index=i,
                        multi_index=_unravel_index(i, sizes),
                        value_a=float(val_a),
                        value_b=float(val_b),
                    )
                )

    mean_abs_diff = total_abs_diff / num_differing if num_differing > 0 else 0.0
    return (count, num_differing, max_abs_diff, mean_abs_diff, element_diffs)


def _make_tensor_info(
    evalue_idx: int,
    tensor: Tensor,
    op_map: Dict[int, List[OperatorUsage]],
) -> TensorInfo:
    fqn = None
    if tensor.extra_tensor_info and tensor.extra_tensor_info.fully_qualified_name:
        fqn = tensor.extra_tensor_info.fully_qualified_name
    return TensorInfo(
        evalue_index=evalue_idx,
        scalar_type=tensor.scalar_type,
        sizes=list(tensor.sizes),
        fqn=fqn,
        data_buffer_idx=tensor.data_buffer_idx,
        operator_usages=op_map.get(evalue_idx, []),
    )


def _diff_tensor_metadata(tensor_a: Tensor, tensor_b: Tensor) -> List[FieldDiff]:
    """Compare all fields of two Tensor dataclasses."""
    diffs: List[FieldDiff] = []
    if tensor_a.scalar_type != tensor_b.scalar_type:
        diffs.append(
            FieldDiff("scalar_type", tensor_a.scalar_type, tensor_b.scalar_type)
        )
    if tensor_a.storage_offset != tensor_b.storage_offset:
        diffs.append(
            FieldDiff(
                "storage_offset", tensor_a.storage_offset, tensor_b.storage_offset
            )
        )
    if tensor_a.sizes != tensor_b.sizes:
        diffs.append(FieldDiff("sizes", tensor_a.sizes, tensor_b.sizes))
    if tensor_a.dim_order != tensor_b.dim_order:
        diffs.append(FieldDiff("dim_order", tensor_a.dim_order, tensor_b.dim_order))
    if tensor_a.requires_grad != tensor_b.requires_grad:
        diffs.append(
            FieldDiff("requires_grad", tensor_a.requires_grad, tensor_b.requires_grad)
        )
    if tensor_a.layout != tensor_b.layout:
        diffs.append(FieldDiff("layout", tensor_a.layout, tensor_b.layout))
    if tensor_a.data_buffer_idx != tensor_b.data_buffer_idx:
        diffs.append(
            FieldDiff(
                "data_buffer_idx", tensor_a.data_buffer_idx, tensor_b.data_buffer_idx
            )
        )
    if tensor_a.allocation_info != tensor_b.allocation_info:
        diffs.append(
            FieldDiff(
                "allocation_info", tensor_a.allocation_info, tensor_b.allocation_info
            )
        )
    if tensor_a.shape_dynamism != tensor_b.shape_dynamism:
        diffs.append(
            FieldDiff(
                "shape_dynamism", tensor_a.shape_dynamism, tensor_b.shape_dynamism
            )
        )
    _diff_extra_tensor_info(
        tensor_a.extra_tensor_info, tensor_b.extra_tensor_info, diffs
    )
    return diffs


def _diff_extra_tensor_info(
    a: Optional[ExtraTensorInfo],
    b: Optional[ExtraTensorInfo],
    diffs: List[FieldDiff],
) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        diffs.append(FieldDiff("extra_tensor_info", a, b))
        return
    if a.mutable_data_segments_idx != b.mutable_data_segments_idx:
        diffs.append(
            FieldDiff(
                "extra_tensor_info.mutable_data_segments_idx",
                a.mutable_data_segments_idx,
                b.mutable_data_segments_idx,
            )
        )
    if a.fully_qualified_name != b.fully_qualified_name:
        diffs.append(
            FieldDiff(
                "extra_tensor_info.fully_qualified_name",
                a.fully_qualified_name,
                b.fully_qualified_name,
            )
        )
    if a.location != b.location:
        diffs.append(
            FieldDiff(
                "extra_tensor_info.location",
                a.location,
                b.location,
            )
        )


def _diff_evalue(  # noqa: C901
    idx: int,
    val_a,
    val_b,
    op_map_a: Dict[int, List[OperatorUsage]],
    op_map_b: Dict[int, List[OperatorUsage]],
) -> Optional[EValueDiff]:
    """Compare two non-Tensor EValue inner values. Returns None if identical."""
    type_a = _EVALUE_TYPE_NAMES.get(type(val_a), type(val_a).__name__)
    type_b = _EVALUE_TYPE_NAMES.get(type(val_b), type(val_b).__name__)

    if type_a != type_b:
        return EValueDiff(
            evalue_index=idx,
            type_a=type_a,
            type_b=type_b,
            type_mismatch=True,
            operator_usages_a=op_map_a.get(idx, []),
            operator_usages_b=op_map_b.get(idx, []),
        )

    field_diffs: List[FieldDiff] = []

    if isinstance(val_a, Null):
        pass
    elif isinstance(val_a, Int):
        if val_a.int_val != val_b.int_val:
            field_diffs.append(FieldDiff("int_val", val_a.int_val, val_b.int_val))
    elif isinstance(val_a, Bool):
        if val_a.bool_val != val_b.bool_val:
            field_diffs.append(FieldDiff("bool_val", val_a.bool_val, val_b.bool_val))
    elif isinstance(val_a, Double):
        if val_a.double_val != val_b.double_val:
            field_diffs.append(
                FieldDiff("double_val", val_a.double_val, val_b.double_val)
            )
    elif isinstance(val_a, String):
        if val_a.string_val != val_b.string_val:
            field_diffs.append(
                FieldDiff("string_val", val_a.string_val, val_b.string_val)
            )
    elif isinstance(val_a, (IntList, TensorList, OptionalTensorList)):
        if val_a.items != val_b.items:
            field_diffs.append(FieldDiff("items", val_a.items, val_b.items))
    elif isinstance(val_a, DoubleList):
        if val_a.items != val_b.items:
            field_diffs.append(FieldDiff("items", val_a.items, val_b.items))
    elif isinstance(val_a, BoolList):
        if val_a.items != val_b.items:
            field_diffs.append(FieldDiff("items", val_a.items, val_b.items))

    if not field_diffs:
        return None

    return EValueDiff(
        evalue_index=idx,
        type_a=type_a,
        type_b=type_b,
        field_diffs=field_diffs,
        operator_usages_a=op_map_a.get(idx, []),
        operator_usages_b=op_map_b.get(idx, []),
    )


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def diff_pte(
    data_a: bytes,
    data_b: bytes,
    path_a: str = "A",
    path_b: str = "B",
    max_samples: int = 10,
) -> PTEDiffResult:
    result = PTEDiffResult(
        path_a=path_a,
        path_b=path_b,
        size_a=len(data_a),
        size_b=len(data_b),
        bitwise_equal=data_a == data_b,
    )

    if result.bitwise_equal:
        return result

    try:
        pte_a = deserialize_pte_binary(data_a)
        pte_b = deserialize_pte_binary(data_b)
    except Exception as e:
        result.error = f"Deserialization failed: {e}"
        return result

    prog_a = pte_a.program
    prog_b = pte_b.program

    if prog_a.version != prog_b.version:
        result.version_a = prog_a.version
        result.version_b = prog_b.version

    num_plans = min(len(prog_a.execution_plan), len(prog_b.execution_plan))

    for i in range(num_plans):
        plan_a = prog_a.execution_plan[i]
        plan_b = prog_b.execution_plan[i]
        plan_diff = _diff_execution_plan(
            i,
            plan_a,
            plan_b,
            prog_a,
            prog_b,
            pte_a.mutable_data,
            pte_b.mutable_data,
            max_samples,
        )
        if plan_diff.has_differences():
            result.plan_diffs.append(plan_diff)

    for i in range(num_plans, len(prog_a.execution_plan)):
        result.extra_plans_in_a.append(prog_a.execution_plan[i].name)
    for i in range(num_plans, len(prog_b.execution_plan)):
        result.extra_plans_in_b.append(prog_b.execution_plan[i].name)

    # Compare named data
    named_a = {nd.key: nd.segment_index for nd in (prog_a.named_data or [])}
    named_b = {nd.key: nd.segment_index for nd in (prog_b.named_data or [])}
    all_keys = set(named_a.keys()) | set(named_b.keys())
    for key in sorted(all_keys):
        if key not in named_a:
            result.named_data_diffs.append(NamedDataDiff(key=key, only_in="B"))
        elif key not in named_b:
            result.named_data_diffs.append(NamedDataDiff(key=key, only_in="A"))

    return result


def _diff_tensors(
    idx: int,
    tensor_a: Tensor,
    tensor_b: Tensor,
    op_map_a: Dict[int, List[OperatorUsage]],
    op_map_b: Dict[int, List[OperatorUsage]],
    constant_buffer_a: list,
    constant_buffer_b: list,
    mutable_data_a: Optional[list],
    mutable_data_b: Optional[list],
    max_samples: int,
) -> Optional[TensorDataDiff]:
    """Compare two tensors at the same evalue index. Returns None if identical."""
    info_a = _make_tensor_info(idx, tensor_a, op_map_a)
    info_b = _make_tensor_info(idx, tensor_b, op_map_b)

    metadata_diffs = _diff_tensor_metadata(tensor_a, tensor_b)

    bytes_a = _get_tensor_bytes(tensor_a, constant_buffer_a, mutable_data_a)
    bytes_b = _get_tensor_bytes(tensor_b, constant_buffer_b, mutable_data_b)

    has_data = bytes_a is not None or bytes_b is not None
    data_matches = bytes_a is not None and bytes_b is not None and bytes_a == bytes_b

    if not metadata_diffs and (not has_data or data_matches):
        return None

    td = TensorDataDiff(
        tensor_a=info_a,
        tensor_b=info_b,
        byte_size_a=len(bytes_a) if bytes_a else 0,
        byte_size_b=len(bytes_b) if bytes_b else 0,
        bytes_differ=not data_matches if has_data else False,
        metadata_diffs=metadata_diffs,
    )

    if has_data and not data_matches and bytes_a is not None and bytes_b is not None:
        (
            td.num_elements,
            td.num_differing,
            td.max_abs_diff,
            td.mean_abs_diff,
            td.element_diffs,
        ) = _compute_tensor_stats(
            bytes_a, bytes_b, tensor_a.scalar_type, tensor_a.sizes, max_samples
        )

    return td


def _diff_execution_plan(  # noqa: C901
    plan_index: int,
    plan_a: ExecutionPlan,
    plan_b: ExecutionPlan,
    prog_a,
    prog_b,
    mutable_data_a: Optional[list],
    mutable_data_b: Optional[list],
    max_samples: int,
) -> ExecutionPlanDiff:
    diff = ExecutionPlanDiff(plan_index=plan_index)

    if plan_a.name != plan_b.name:
        diff.name_a = plan_a.name
        diff.name_b = plan_b.name

    # Compare operators (multiset diff)
    ops_a = Counter(
        f"{op.name}.{op.overload}" if op.overload else op.name
        for op in plan_a.operators
    )
    ops_b = Counter(
        f"{op.name}.{op.overload}" if op.overload else op.name
        for op in plan_b.operators
    )
    if ops_a != ops_b:
        only_a = ops_a - ops_b
        only_b = ops_b - ops_a
        diff.operators_only_in_a = sorted(only_a.elements())
        diff.operators_only_in_b = sorted(only_b.elements())

    # Compare delegates
    delegates_a = Counter(d.id for d in plan_a.delegates)
    delegates_b = Counter(d.id for d in plan_b.delegates)
    if delegates_a != delegates_b:
        only_a = delegates_a - delegates_b
        only_b = delegates_b - delegates_a
        diff.delegates_only_in_a = sorted(only_a.elements())
        diff.delegates_only_in_b = sorted(only_b.elements())

    # Compare non-const buffer sizes
    if plan_a.non_const_buffer_sizes != plan_b.non_const_buffer_sizes:
        diff.non_const_buffer_sizes_a = plan_a.non_const_buffer_sizes
        diff.non_const_buffer_sizes_b = plan_b.non_const_buffer_sizes

    # Compare value count
    num_values_a = len(plan_a.values)
    num_values_b = len(plan_b.values)
    if num_values_a != num_values_b:
        diff.value_count_a = num_values_a
        diff.value_count_b = num_values_b

    # Compare instruction count
    instr_count_a = sum(len(c.instructions) for c in plan_a.chains)
    instr_count_b = sum(len(c.instructions) for c in plan_b.chains)
    if instr_count_a != instr_count_b:
        diff.instruction_count_a = instr_count_a
        diff.instruction_count_b = instr_count_b

    op_map_a = _build_tensor_to_operators(plan_a)
    op_map_b = _build_tensor_to_operators(plan_b)

    # Compare values
    num_values = min(num_values_a, num_values_b)
    for idx in range(num_values):
        ev_a = plan_a.values[idx]
        ev_b = plan_b.values[idx]

        is_tensor_a = isinstance(ev_a.val, Tensor)
        is_tensor_b = isinstance(ev_b.val, Tensor)

        # Both are tensors: full tensor comparison
        if is_tensor_a and is_tensor_b:
            td = _diff_tensors(
                idx,
                ev_a.val,
                ev_b.val,
                op_map_a,
                op_map_b,
                prog_a.constant_buffer,
                prog_b.constant_buffer,
                mutable_data_a,
                mutable_data_b,
                max_samples,
            )
            if td is not None:
                diff.tensor_diffs.append(td)
            continue

        # Non-tensor value comparison (including type mismatch)
        evd = _diff_evalue(idx, ev_a.val, ev_b.val, op_map_a, op_map_b)
        if evd is not None:
            diff.evalue_diffs.append(evd)

    # Report extra tensors/values from the longer side
    for idx in range(num_values, num_values_a):
        ev = plan_a.values[idx]
        if isinstance(ev.val, Tensor):
            diff.tensors_only_in_a.append(_make_tensor_info(idx, ev.val, op_map_a))
    for idx in range(num_values, num_values_b):
        ev = plan_b.values[idx]
        if isinstance(ev.val, Tensor):
            diff.tensors_only_in_b.append(_make_tensor_info(idx, ev.val, op_map_b))

    return diff


def _format_tensor_info(info: TensorInfo) -> str:
    kind = "CT" if info.data_buffer_idx > 0 else "T"
    dtype = _SCALAR_TYPE_NAMES.get(info.scalar_type, str(info.scalar_type))
    sizes_str = ",".join(str(s) for s in info.sizes)
    fqn_str = f'  "{info.fqn}"' if info.fqn else ""
    return f"[idx={info.evalue_index}]  {kind} {dtype}[{sizes_str}]{fqn_str}"


def format_diff_result(  # noqa: C901
    result: PTEDiffResult, verbose: bool = False
) -> str:
    lines: List[str] = []

    lines.append(
        f"Comparing: {result.path_a} ({_format_size(result.size_a)}) "
        f"vs {result.path_b} ({_format_size(result.size_b)})"
    )

    if result.bitwise_equal:
        lines.append("Status: bitwise equal")
        return "\n".join(lines)

    lines.append("Status: NOT bitwise equal")

    if result.error:
        lines.append(f"Error: {result.error}")
        return "\n".join(lines)

    if result.version_a is not None:
        lines.append(f"Version: {result.version_a} vs {result.version_b}")

    for pd in result.plan_diffs:
        plan_name = pd.name_a or pd.name_b or ""
        header = f"Plan {pd.plan_index}"
        if plan_name:
            header += f' "{plan_name}"'
        if pd.name_a is not None and pd.name_b is not None:
            header += f" (name differs: {pd.name_a!r} vs {pd.name_b!r})"
        lines.append(f"\n{header}:")

        if pd.operators_only_in_a:
            lines.append(f"  Operators only in A: {', '.join(pd.operators_only_in_a)}")
        if pd.operators_only_in_b:
            lines.append(f"  Operators only in B: {', '.join(pd.operators_only_in_b)}")

        if pd.delegates_only_in_a:
            lines.append(f"  Delegates only in A: {', '.join(pd.delegates_only_in_a)}")
        if pd.delegates_only_in_b:
            lines.append(f"  Delegates only in B: {', '.join(pd.delegates_only_in_b)}")

        if pd.non_const_buffer_sizes_a is not None:
            lines.append(
                f"  non_const_buffer_sizes: {pd.non_const_buffer_sizes_a} "
                f"vs {pd.non_const_buffer_sizes_b}"
            )

        if pd.value_count_a is not None:
            lines.append(f"  Value count: {pd.value_count_a} vs {pd.value_count_b}")

        if pd.instruction_count_a is not None:
            lines.append(
                f"  Instruction count: {pd.instruction_count_a} "
                f"vs {pd.instruction_count_b}"
            )

        if pd.tensor_diffs:
            lines.append(f"  Differing tensors ({len(pd.tensor_diffs)}):")
            for td in pd.tensor_diffs:
                lines.append(f"    {_format_tensor_info(td.tensor_a)}")
                if verbose:
                    lines.append(
                        f"      byte sizes: {td.byte_size_a} vs {td.byte_size_b}"
                    )
                if td.tensor_a.operator_usages:
                    for usage in td.tensor_a.operator_usages:
                        lines.append(
                            f"      used as arg {usage.arg_index} of {usage.name}"
                        )
                elif verbose and td.tensor_b.operator_usages:
                    for usage in td.tensor_b.operator_usages:
                        lines.append(
                            f"      used as arg {usage.arg_index} of {usage.name}"
                        )
                if td.metadata_diffs:
                    for md in td.metadata_diffs:
                        lines.append(
                            f"      {md.field_name}: {md.value_a} vs {md.value_b}"
                        )
                if td.bytes_differ and td.num_differing > 0:
                    lines.append(
                        f"      {td.num_differing} / {td.num_elements} "
                        f"elements differ"
                    )
                    lines.append(
                        f"      max_abs_diff={td.max_abs_diff:.6g}  "
                        f"mean_abs_diff={td.mean_abs_diff:.6g}"
                    )
                    if td.element_diffs:
                        lines.append("      first differences:")
                        for ed in td.element_diffs:
                            idx_str = ",".join(str(x) for x in ed.multi_index)
                            diff_val = abs(ed.value_a - ed.value_b)
                            lines.append(
                                f"        [{idx_str}]: {ed.value_a} vs "
                                f"{ed.value_b} (diff={diff_val:.6g})"
                            )
                        remaining = td.num_differing - len(td.element_diffs)
                        if remaining > 0:
                            lines.append(f"        ... and {remaining} more")

        if pd.evalue_diffs:
            lines.append(f"  Differing values ({len(pd.evalue_diffs)}):")
            for evd in pd.evalue_diffs:
                if evd.type_mismatch:
                    lines.append(
                        f"    [idx={evd.evalue_index}]  type: "
                        f"{evd.type_a} vs {evd.type_b}"
                    )
                else:
                    lines.append(f"    [idx={evd.evalue_index}]  {evd.type_a}")
                    for fd in evd.field_diffs:
                        lines.append(
                            f"      {fd.field_name}: " f"{fd.value_a} vs {fd.value_b}"
                        )
                if evd.operator_usages_a:
                    for usage in evd.operator_usages_a:
                        lines.append(
                            f"      used in A as arg {usage.arg_index} "
                            f"of {usage.name}"
                        )
                if (
                    evd.operator_usages_b
                    and evd.operator_usages_b != evd.operator_usages_a
                ):
                    for usage in evd.operator_usages_b:
                        lines.append(
                            f"      used in B as arg {usage.arg_index} "
                            f"of {usage.name}"
                        )

        if pd.tensors_only_in_a:
            lines.append(f"  Tensors only in A ({len(pd.tensors_only_in_a)}):")
            for info in pd.tensors_only_in_a:
                lines.append(f"    {_format_tensor_info(info)}")

        if pd.tensors_only_in_b:
            lines.append(f"  Tensors only in B ({len(pd.tensors_only_in_b)}):")
            for info in pd.tensors_only_in_b:
                lines.append(f"    {_format_tensor_info(info)}")

    if result.extra_plans_in_a:
        lines.append(f"\nExtra plans in A: {', '.join(result.extra_plans_in_a)}")
    if result.extra_plans_in_b:
        lines.append(f"\nExtra plans in B: {', '.join(result.extra_plans_in_b)}")

    if result.named_data_diffs:
        lines.append("\nNamed data differences:")
        for nd in result.named_data_diffs:
            lines.append(f"  {nd.key!r}: only in {nd.only_in}")

    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ExecuTorch .pte files")
    parser.add_argument("file_a", help="First .pte file")
    parser.add_argument("file_b", help="Second .pte file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show additional details"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Max number of per-tensor element diffs to show (default: 10)",
    )
    args = parser.parse_args()

    with open(args.file_a, "rb") as f:
        data_a = f.read()
    with open(args.file_b, "rb") as f:
        data_b = f.read()

    result = diff_pte(
        data_a, data_b, args.file_a, args.file_b, max_samples=args.max_samples
    )
    print(format_diff_result(result, verbose=args.verbose))


if __name__ == "__main__":
    _main()
