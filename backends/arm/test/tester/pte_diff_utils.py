# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
from collections import Counter
from math import isclose
from numbers import Real
from typing import Any, Callable, Dict, Tuple

from executorch.devtools.pte_tool.diff_pte import PTEDiffResult
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import DataLocation, Tensor

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


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _numeric_values_close(
    value_a: Any,
    value_b: Any,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if (
        isinstance(value_a, Real)
        and not isinstance(value_a, bool)
        and isinstance(value_b, Real)
        and not isinstance(value_b, bool)
    ):
        return isclose(value_a, value_b, abs_tol=atol, rel_tol=rtol)

    if isinstance(value_a, list) and isinstance(value_b, list):
        return len(value_a) == len(value_b) and all(
            _numeric_values_close(item_a, item_b, atol=atol, rtol=rtol)
            for item_a, item_b in zip(value_a, value_b)
        )

    return False


def _tensor_bytes(pte, plan_index: int, evalue_index: int) -> bytes | None:
    tensor = pte.program.execution_plan[plan_index].values[evalue_index].val
    if tensor.data_buffer_idx > 0 and tensor.data_buffer_idx < len(
        pte.program.constant_buffer
    ):
        return pte.program.constant_buffer[tensor.data_buffer_idx].storage
    mutable_data = getattr(pte, "mutable_data", None)
    if (
        tensor.extra_tensor_info
        and tensor.extra_tensor_info.mutable_data_segments_idx > 0
        and mutable_data is not None
        and tensor.extra_tensor_info.mutable_data_segments_idx < len(mutable_data)
    ):
        return mutable_data[tensor.extra_tensor_info.mutable_data_segments_idx].storage
    return None


def _tensor_payload_within_tolerance(
    *,
    data_a: bytes,
    data_b: bytes,
    scalar_type: ScalarType,
    storage_offset_a: int,
    storage_offset_b: int,
    num_elements: int,
    atol: float,
    rtol: float,
) -> bool:
    fmt_info = _SCALAR_TYPE_TO_FORMAT.get(scalar_type)
    if fmt_info is None:
        return False

    fmt_char, elem_size = fmt_info
    if len(data_a) != len(data_b):
        return False

    start_a = storage_offset_a * elem_size
    start_b = storage_offset_b * elem_size
    end_a = start_a + num_elements * elem_size
    end_b = start_b + num_elements * elem_size
    if start_a < 0 or start_b < 0 or end_a > len(data_a) or end_b > len(data_b):
        return False

    saw_element_bytes_differ = False
    for index in range(num_elements):
        offset_a = start_a + index * elem_size
        offset_b = start_b + index * elem_size
        if (
            data_a[offset_a : offset_a + elem_size]
            != data_b[offset_b : offset_b + elem_size]
        ):
            saw_element_bytes_differ = True
        (value_a,) = struct.unpack_from(f"<{fmt_char}", data_a, offset_a)
        (value_b,) = struct.unpack_from(f"<{fmt_char}", data_b, offset_b)
        if value_a != value_b and not _numeric_values_close(
            value_a,
            value_b,
            atol=atol,
            rtol=rtol,
        ):
            return False

    return saw_element_bytes_differ


def _referenced_backend_delegate_payloads(program) -> list[tuple[str, bytes | None]]:
    payloads: list[tuple[str, bytes | None]] = []
    delegate_data = program.backend_delegate_data or []
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            if delegate.processed.location != DataLocation.INLINE:
                payloads.append((delegate.id, None))
                continue

            index = delegate.processed.index
            if index < 0 or index >= len(delegate_data):
                payloads.append((delegate.id, None))
                continue

            payloads.append((delegate.id, delegate_data[index].data))
    return payloads


def pte_diff_within_tolerance(  # noqa: C901
    result: PTEDiffResult,
    data_a: bytes,
    data_b: bytes,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if atol == 0.0 and rtol == 0.0:
        return False

    if (
        result.error
        or result.version_a is not None
        or result.size_a != result.size_b
        or result.extra_plans_in_a
        or result.extra_plans_in_b
        or result.named_data_diffs
    ):
        return False

    saw_tolerated_diff = False
    pte_a = None
    pte_b = None
    for plan_diff in result.plan_diffs:
        if (
            plan_diff.name_a is not None
            or plan_diff.operators_only_in_a
            or plan_diff.operators_only_in_b
            or plan_diff.delegates_only_in_a
            or plan_diff.delegates_only_in_b
            or plan_diff.non_const_buffer_sizes_a is not None
            or plan_diff.value_count_a is not None
            or plan_diff.instruction_count_a is not None
            or plan_diff.tensors_only_in_a
            or plan_diff.tensors_only_in_b
        ):
            return False

        for tensor_diff in plan_diff.tensor_diffs:
            if pte_a is None or pte_b is None:
                pte_a = deserialize_pte_binary(data_a)
                pte_b = deserialize_pte_binary(data_b)

            tensor_a = (
                pte_a.program.execution_plan[plan_diff.plan_index]
                .values[tensor_diff.tensor_a.evalue_index]
                .val
            )
            tensor_b = (
                pte_b.program.execution_plan[plan_diff.plan_index]
                .values[tensor_diff.tensor_b.evalue_index]
                .val
            )
            if not isinstance(tensor_a, Tensor) or not isinstance(tensor_b, Tensor):
                return False
            tensor_bytes_a = _tensor_bytes(
                pte_a, plan_diff.plan_index, tensor_diff.tensor_a.evalue_index
            )
            tensor_bytes_b = _tensor_bytes(
                pte_b, plan_diff.plan_index, tensor_diff.tensor_b.evalue_index
            )
            if (
                tensor_diff.metadata_diffs
                or not tensor_diff.bytes_differ
                or tensor_diff.byte_size_a != tensor_diff.byte_size_b
                or tensor_bytes_a is None
                or tensor_bytes_b is None
                or not _tensor_payload_within_tolerance(
                    data_a=tensor_bytes_a,
                    data_b=tensor_bytes_b,
                    scalar_type=tensor_diff.tensor_a.scalar_type,
                    storage_offset_a=tensor_a.storage_offset,
                    storage_offset_b=tensor_b.storage_offset,
                    num_elements=tensor_diff.num_elements,
                    atol=atol,
                    rtol=rtol,
                )
            ):
                return False
            saw_tolerated_diff = True

        for evalue_diff in plan_diff.evalue_diffs:
            if (
                evalue_diff.type_mismatch
                or evalue_diff.type_a != evalue_diff.type_b
                or evalue_diff.type_a not in {"Double", "DoubleList"}
            ):
                return False

            for field_diff in evalue_diff.field_diffs:
                if field_diff.field_name not in {"double_val", "items"}:
                    return False
                if not _numeric_values_close(
                    field_diff.value_a,
                    field_diff.value_b,
                    atol=atol,
                    rtol=rtol,
                ):
                    return False
            saw_tolerated_diff = True

    if not saw_tolerated_diff:
        return False

    if pte_a is None or pte_b is None:
        pte_a = deserialize_pte_binary(data_a)
        pte_b = deserialize_pte_binary(data_b)

    return _referenced_backend_delegate_payloads(
        pte_a.program
    ) == _referenced_backend_delegate_payloads(pte_b.program)


def _tosa_delegate_payloads(program) -> list[bytes]:
    payloads: list[bytes] = []
    delegate_data = program.backend_delegate_data or []
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            if delegate.id != "TOSABackend":
                continue
            if delegate.processed.location != DataLocation.INLINE:
                continue
            index = delegate.processed.index
            if index < 0 or index >= len(delegate_data):
                continue
            payloads.append(delegate_data[index].data)
    return payloads


def _tosa_op_names() -> dict[int, str]:
    from tosa.Op import Op  # type: ignore[import-untyped]

    return {
        value: name
        for name, value in vars(Op).items()
        if name.isupper() and isinstance(value, int)
    }


def _read_tosa_ops(data: bytes) -> list[str]:
    from tosa.TosaGraph import TosaGraph  # type: ignore[import-untyped]

    op_names = _tosa_op_names()
    graph = TosaGraph.GetRootAsTosaGraph(data, 0)
    ops: list[str] = []
    for region_idx in range(graph.RegionsLength()):
        region = graph.Regions(region_idx)
        for block_idx in range(region.BlocksLength()):
            block = region.Blocks(block_idx)
            for op_idx in range(block.OperatorsLength()):
                op_code = block.Operators(op_idx).Op()
                ops.append(op_names.get(op_code, str(op_code)))
    return ops


def _summarize_tosa_ops(
    ops_a: list[str],
    ops_b: list[str],
    max_samples: int,
) -> list[str]:
    lines = [f"    TOSA op count: {len(ops_a)} vs {len(ops_b)}"]

    only_a = sorted((Counter(ops_a) - Counter(ops_b)).elements())
    only_b = sorted((Counter(ops_b) - Counter(ops_a)).elements())
    if only_a:
        lines.append(f"    TOSA ops only in A: {', '.join(only_a[:max_samples])}")
    if only_b:
        lines.append(f"    TOSA ops only in B: {', '.join(only_b[:max_samples])}")

    for op_idx, (op_a, op_b) in enumerate(zip(ops_a, ops_b)):
        if op_a != op_b:
            lines.append(
                f"    TOSA operator type mismatch: op[{op_idx}]: {op_a} vs {op_b}"
            )
            break
    else:
        if len(ops_a) != len(ops_b):
            lines.append(
                f"    TOSA operator type mismatch: op[{min(len(ops_a), len(ops_b))}] "
                "exists only in one PTE"
            )
        else:
            lines.append(f"    TOSA ops match ({len(ops_a)} ops)")

    return lines


def tosa_delegate_payloads_equal(data_a: bytes, data_b: bytes) -> bool:
    try:
        prog_a = deserialize_pte_binary(data_a).program
        prog_b = deserialize_pte_binary(data_b).program
    except Exception:
        return False

    payloads_a = _tosa_delegate_payloads(prog_a)
    return bool(payloads_a) and payloads_a == _tosa_delegate_payloads(prog_b)


def tosa_delegate_ops_equal(
    data_a: bytes,
    data_b: bytes,
    read_tosa_ops: Callable[[bytes], list[str]] = _read_tosa_ops,
) -> bool:
    try:
        prog_a = deserialize_pte_binary(data_a).program
        prog_b = deserialize_pte_binary(data_b).program
    except Exception:
        return False

    payloads_a = _tosa_delegate_payloads(prog_a)
    payloads_b = _tosa_delegate_payloads(prog_b)
    if not payloads_a or len(payloads_a) != len(payloads_b):
        return False
    if payloads_a == payloads_b:
        return True

    try:
        return all(
            read_tosa_ops(payload_a) == read_tosa_ops(payload_b)
            for payload_a, payload_b in zip(payloads_a, payloads_b)
        )
    except Exception:
        return False


def format_tosa_delegate_diff(
    data_a: bytes,
    data_b: bytes,
    max_samples: int,
    read_tosa_ops: Callable[[bytes], list[str]] = _read_tosa_ops,
) -> list[str]:
    try:
        prog_a = deserialize_pte_binary(data_a).program
        prog_b = deserialize_pte_binary(data_b).program
    except Exception as ex:
        return [f"TOSA op diff unavailable: PTE deserialization failed: {ex}"]

    tosa_payloads_a = _tosa_delegate_payloads(prog_a)
    tosa_payloads_b = _tosa_delegate_payloads(prog_b)
    num_delegate_data = max(len(tosa_payloads_a), len(tosa_payloads_b))
    if num_delegate_data == 0:
        return []

    lines = ["Backend delegate TOSA op differences:"]
    for idx in range(num_delegate_data):
        if idx >= len(tosa_payloads_a):
            lines.append(
                f"  [{idx}]: only in B ({_format_size(len(tosa_payloads_b[idx]))})"
            )
            continue
        if idx >= len(tosa_payloads_b):
            lines.append(
                f"  [{idx}]: only in A ({_format_size(len(tosa_payloads_a[idx]))})"
            )
            continue

        payload_a = tosa_payloads_a[idx]
        payload_b = tosa_payloads_b[idx]
        lines.append(
            f"  [{idx}]: {_format_size(len(payload_a))} vs {_format_size(len(payload_b))}"
        )
        try:
            lines.extend(
                _summarize_tosa_ops(
                    read_tosa_ops(payload_a),
                    read_tosa_ops(payload_b),
                    max_samples,
                )
            )
        except Exception as ex:
            lines.append(f"    TOSA op diff unavailable: {ex}")

    return lines if len(lines) > 1 else []
