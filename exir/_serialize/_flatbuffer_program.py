# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import importlib.resources
import re

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import flatbuffers  # pyre-ignore[21]

from executorch.exir._serialize._flatbuffer import (
    _FlatbufferResult,
    _is_valid_alignment,
    _patch_schema_alignment,
    _ResourceFiles,
    _SchemaMaxAlignmentGetter,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    AllocationDetails,
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Bool,
    BoolList,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    DataSegment,
    DelegateCall,
    Double,
    DoubleList,
    EValue,
    ExecutionPlan,
    ExtraTensorInfo,
    Frame,
    FrameList,
    FreeCall,
    Instruction,
    Int,
    IntList,
    JumpFalseCall,
    KernelCall,
    MoveCall,
    NamedData,
    Null,
    Operator,
    OptionalTensorList,
    Program,
    String,
    SubsegmentOffsets,
    Tensor,
    TensorDataLocation,
    TensorList,
    TensorShapeDynamism,
)


def _read_resource_bytes(name: str) -> bytes:
    return importlib.resources.files(__package__).joinpath(name).read_bytes()


def _get_schema_details(
    *,
    constant_tensor_alignment: Optional[int],
    delegate_alignment: Optional[int],
) -> Tuple[bytes, int, int, int]:
    """Returns (file_identifier, tensor_alignment, delegate_alignment, max_alignment).

    The returned alignments reflect any provided override values, otherwise
    they come from the schema defaults.
    """
    # Validate and apply alignment overrides so we can inspect the effective
    # values from the (possibly-patched) schema.
    program_schema: bytes = _read_resource_bytes("program.fbs")
    patched_program_schema: bytes = _patch_schema_alignment(
        schema=program_schema,
        constant_tensor_alignment=constant_tensor_alignment,
        delegate_alignment=delegate_alignment,
    )

    def extract_file_identifier(schema: bytes) -> bytes:
        match = re.search(rb'file_identifier\s*"([^"]{4})"\s*;', schema)
        if not match:
            raise RuntimeError("Failed to find file_identifier in program.fbs")
        return match.group(1)

    def extract_alignment(schema: bytes, marker: bytes) -> int:
        for line in schema.splitlines():
            if marker in line:
                match = re.search(rb"force_align\s*:\s*(\d+)", line)
                if match:
                    return int(match.group(1))
        raise RuntimeError(f"Failed to find marker {marker!r} in program.fbs")

    file_identifier: bytes = extract_file_identifier(patched_program_schema)
    tensor_alignment: int = extract_alignment(
        patched_program_schema, b"@executorch-tensor-alignment"
    )
    effective_delegate_alignment: int = extract_alignment(
        patched_program_schema, b"@executorch-delegate-alignment"
    )

    # Compute the max force_align across all schema files (after patching).
    program_schema_name = "program.fbs"
    deps = ["scalar_type.fbs"]
    schemas = _ResourceFiles([program_schema_name] + deps)
    schemas.patch_files(
        lambda data: _patch_schema_alignment(
            schema=data,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
    )
    get_alignments = _SchemaMaxAlignmentGetter()
    schemas.patch_files(get_alignments)
    max_alignment: int = get_alignments.max_alignment
    if max_alignment <= 0:
        raise RuntimeError(f"Invalid max_alignment {max_alignment}")

    return (
        file_identifier,
        tensor_alignment,
        effective_delegate_alignment,
        max_alignment,
    )


@functools.lru_cache(maxsize=1)
def _get_union_type_ids() -> Tuple[Dict[str, int], Dict[str, int]]:
    """Returns (KernelTypes enum map, InstructionArguments enum map).

    Each map is from type name to the numeric union discriminator.
    """
    schema: bytes = _read_resource_bytes("program.fbs")

    def parse_union(union_name: str) -> Dict[str, int]:
        pattern = rb"union\s+" + union_name.encode("ascii") + rb"\s*\{(.*?)\}"
        match = re.search(pattern, schema, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Failed to find union {union_name} in program.fbs")
        body: bytes = match.group(1)
        # Drop line and block comments.
        body = re.sub(rb"//.*", b"", body)
        body = re.sub(rb"/\*.*?\*/", b"", body, flags=re.DOTALL)
        names = [name.strip() for name in body.split(b",") if name.strip()]
        # Union discriminators start at 1; 0 is reserved for NONE.
        return {name.decode("ascii"): i + 1 for i, name in enumerate(names)}

    return parse_union("KernelTypes"), parse_union("InstructionArguments")


def _create_uoffset_vector(builder: Any, offsets: Sequence[int]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        builder.PrependUOffsetTRelative(off)
    return builder.EndVector()


def _create_int32_vector(builder: Any, values: Sequence[int]) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(int(value))
    return builder.EndVector()


def _create_int64_vector(builder: Any, values: Sequence[int]) -> int:
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependInt64(int(value))
    return builder.EndVector()


def _create_uint64_vector(builder: Any, values: Sequence[int]) -> int:
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependUint64(int(value))
    return builder.EndVector()


def _create_uint8_vector(builder: Any, values: Sequence[int]) -> int:
    builder.StartVector(1, len(values), 1)
    for value in reversed(values):
        builder.PrependUint8(int(value))
    return builder.EndVector()


def _create_bool_vector(builder: Any, values: Sequence[bool]) -> int:
    builder.StartVector(1, len(values), 1)
    for value in reversed(values):
        builder.PrependBool(bool(value))
    return builder.EndVector()


def _create_float64_vector(builder: Any, values: Sequence[float]) -> int:
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependFloat64(float(value))
    return builder.EndVector()


def _create_aligned_byte_vector(builder: Any, data: bytes, alignment: int) -> int:
    if not _is_valid_alignment(alignment):
        raise ValueError(f"Bad alignment {alignment}")
    builder.StartVector(1, len(data), alignment)
    # Efficiently write the raw bytes payload (avoids per-byte PrependUint8).
    length = len(data)
    builder.head = builder.Head() - length  # pyre-ignore[16]
    builder.Bytes[builder.Head() : builder.Head() + length] = data  # pyre-ignore[16]
    return builder.EndVector()


def _build_null(builder: Any, _: Null) -> int:
    builder.StartObject(0)
    return builder.EndObject()


def _build_int(builder: Any, val: Int) -> int:
    builder.StartObject(1)
    builder.PrependInt64Slot(0, int(val.int_val), 0)
    return builder.EndObject()


def _build_bool(builder: Any, val: Bool) -> int:
    builder.StartObject(1)
    builder.PrependBoolSlot(0, bool(val.bool_val), False)
    return builder.EndObject()


def _build_double(builder: Any, val: Double) -> int:
    double_val = val.double_val
    if isinstance(double_val, str):
        if double_val == "inf":
            double_num = float("inf")
        elif double_val == "-inf":
            double_num = float("-inf")
        else:
            raise ValueError(f"Unexpected Double string value {double_val!r}")
    else:
        double_num = float(double_val)
    builder.StartObject(1)
    builder.PrependFloat64Slot(0, double_num, 0.0)
    return builder.EndObject()


def _build_string(builder: Any, val: String) -> int:
    string_off = builder.CreateString(val.string_val)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, string_off, 0)
    return builder.EndObject()


def _build_int_list(builder: Any, val: IntList) -> int:
    items_off = _create_int64_vector(builder, val.items)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_double_list(builder: Any, val: DoubleList) -> int:
    items_off = _create_float64_vector(builder, val.items)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_bool_list(builder: Any, val: BoolList) -> int:
    items_off = _create_bool_vector(builder, val.items)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_tensor_list(builder: Any, val: TensorList) -> int:
    items_off = _create_int32_vector(builder, val.items)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_optional_tensor_list(builder: Any, val: OptionalTensorList) -> int:
    items_off = _create_int32_vector(builder, val.items)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_container_metadata(builder: Any, meta: ContainerMetadata) -> int:
    inp_off = builder.CreateString(meta.encoded_inp_str)
    out_off = builder.CreateString(meta.encoded_out_str)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, inp_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, out_off, 0)
    return builder.EndObject()


def _build_allocation_details(builder: Any, val: AllocationDetails) -> int:
    builder.StartObject(3)
    builder.PrependUint32Slot(0, int(val.memory_id), 0)
    builder.PrependUint32Slot(1, int(val.memory_offset_low), 0)
    builder.PrependUint32Slot(2, int(val.memory_offset_high), 0)
    return builder.EndObject()


def _build_extra_tensor_info(builder: Any, val: ExtraTensorInfo) -> int:
    fqn_off = (
        builder.CreateString(val.fully_qualified_name)
        if val.fully_qualified_name is not None
        else 0
    )
    builder.StartObject(3)
    builder.PrependUint64Slot(0, int(val.mutable_data_segments_idx), 0)
    if fqn_off != 0:
        builder.PrependUOffsetTRelativeSlot(1, fqn_off, 0)
    builder.PrependInt8Slot(2, int(val.location), int(TensorDataLocation.SEGMENT))
    return builder.EndObject()


def _build_tensor(builder: Any, val: Tensor) -> int:
    sizes_off = _create_int32_vector(builder, val.sizes)
    dim_order_off = _create_uint8_vector(builder, val.dim_order)
    allocation_info_off = (
        _build_allocation_details(builder, val.allocation_info)
        if val.allocation_info is not None
        else 0
    )
    extra_info_off = (
        _build_extra_tensor_info(builder, val.extra_tensor_info)
        if val.extra_tensor_info is not None
        else 0
    )
    builder.StartObject(10)
    builder.PrependInt8Slot(0, int(val.scalar_type), 0)
    builder.PrependInt32Slot(1, int(val.storage_offset), 0)
    builder.PrependUOffsetTRelativeSlot(2, sizes_off, 0)
    builder.PrependUOffsetTRelativeSlot(3, dim_order_off, 0)
    builder.PrependBoolSlot(4, bool(val.requires_grad), False)
    builder.PrependUint32Slot(5, int(val.data_buffer_idx), 0)
    if allocation_info_off != 0:
        builder.PrependUOffsetTRelativeSlot(6, allocation_info_off, 0)
    builder.PrependInt8Slot(7, int(val.layout), 0)
    builder.PrependInt8Slot(8, int(val.shape_dynamism), int(TensorShapeDynamism.STATIC))
    if extra_info_off != 0:
        builder.PrependUOffsetTRelativeSlot(9, extra_info_off, 0)
    return builder.EndObject()


_EVALUE_BUILDERS: Dict[Type[Any], Callable[[Any, Any], int]] = {
    Null: _build_null,
    Int: _build_int,
    Bool: _build_bool,
    Double: _build_double,
    Tensor: _build_tensor,
    String: _build_string,
    IntList: _build_int_list,
    DoubleList: _build_double_list,
    BoolList: _build_bool_list,
    TensorList: _build_tensor_list,
    OptionalTensorList: _build_optional_tensor_list,
}


def _build_evalue(
    builder: Any, val: EValue, *, kernel_union_ids: Dict[str, int]
) -> int:
    union_val = val.val
    type_name = type(union_val).__name__
    type_id = kernel_union_ids.get(type_name)
    if type_id is None:
        raise ValueError(f"Unsupported KernelTypes variant {type_name}")

    builder_fn = _EVALUE_BUILDERS.get(type(union_val))
    if builder_fn is None:
        raise ValueError(f"Unsupported KernelTypes value {union_val!r}")
    obj_off = builder_fn(builder, union_val)

    builder.StartObject(2)
    builder.PrependUint8Slot(0, int(type_id), 0)
    builder.PrependUOffsetTRelativeSlot(1, obj_off, 0)
    return builder.EndObject()


def _build_operator(builder: Any, val: Operator) -> int:
    name_off = builder.CreateString(val.name)
    overload_off = builder.CreateString(val.overload)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, name_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, overload_off, 0)
    return builder.EndObject()


def _build_kernel_call(builder: Any, val: KernelCall) -> int:
    args_off = _create_int32_vector(builder, val.args)
    builder.StartObject(2)
    builder.PrependInt32Slot(0, int(val.op_index), 0)
    builder.PrependUOffsetTRelativeSlot(1, args_off, 0)
    return builder.EndObject()


def _build_delegate_call(builder: Any, val: DelegateCall) -> int:
    args_off = _create_int32_vector(builder, val.args)
    builder.StartObject(2)
    builder.PrependInt32Slot(0, int(val.delegate_index), 0)
    builder.PrependUOffsetTRelativeSlot(1, args_off, 0)
    return builder.EndObject()


def _build_move_call(builder: Any, val: MoveCall) -> int:
    builder.StartObject(2)
    builder.PrependInt32Slot(0, int(val.move_from), 0)
    builder.PrependInt32Slot(1, int(val.move_to), 0)
    return builder.EndObject()


def _build_jump_false_call(builder: Any, val: JumpFalseCall) -> int:
    builder.StartObject(2)
    builder.PrependInt32Slot(0, int(val.cond_value_index), 0)
    builder.PrependInt32Slot(1, int(val.destination_instruction), 0)
    return builder.EndObject()


def _build_free_call(builder: Any, val: FreeCall) -> int:
    builder.StartObject(1)
    builder.PrependInt32Slot(0, int(val.value_index), 0)
    return builder.EndObject()


def _build_instruction(
    builder: Any, val: Instruction, *, instr_union_ids: Dict[str, int]
) -> int:
    union_val = val.instr_args
    type_name = type(union_val).__name__
    type_id = instr_union_ids.get(type_name)
    if type_id is None:
        raise ValueError(f"Unsupported InstructionArguments variant {type_name}")

    if isinstance(union_val, KernelCall):
        obj_off = _build_kernel_call(builder, union_val)
    elif isinstance(union_val, DelegateCall):
        obj_off = _build_delegate_call(builder, union_val)
    elif isinstance(union_val, MoveCall):
        obj_off = _build_move_call(builder, union_val)
    elif isinstance(union_val, JumpFalseCall):
        obj_off = _build_jump_false_call(builder, union_val)
    elif isinstance(union_val, FreeCall):
        obj_off = _build_free_call(builder, union_val)
    else:
        raise ValueError(f"Unsupported InstructionArguments value {union_val!r}")

    builder.StartObject(2)
    builder.PrependUint8Slot(0, int(type_id), 0)
    builder.PrependUOffsetTRelativeSlot(1, obj_off, 0)
    return builder.EndObject()


def _build_frame(builder: Any, val: Frame) -> int:
    filename_off = builder.CreateString(val.filename)
    name_off = builder.CreateString(val.name)
    context_off = builder.CreateString(val.context)
    builder.StartObject(4)
    builder.PrependUOffsetTRelativeSlot(0, filename_off, 0)
    builder.PrependInt32Slot(1, int(val.lineno), 0)
    builder.PrependUOffsetTRelativeSlot(2, name_off, 0)
    builder.PrependUOffsetTRelativeSlot(3, context_off, 0)
    return builder.EndObject()


def _build_frame_list(builder: Any, val: FrameList) -> int:
    frame_offsets = [_build_frame(builder, frame) for frame in val.items]
    items_off = _create_uoffset_vector(builder, frame_offsets)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, items_off, 0)
    return builder.EndObject()


def _build_backend_delegate_data_reference(
    builder: Any, val: BackendDelegateDataReference
) -> int:
    builder.StartObject(2)
    builder.PrependInt8Slot(0, int(val.location), int(DataLocation.INLINE))
    builder.PrependUint32Slot(1, int(val.index), 0)
    return builder.EndObject()


def _build_compile_spec(builder: Any, val: CompileSpec) -> int:
    key_off = builder.CreateString(val.key)
    value_off = builder.CreateByteVector(val.value)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, key_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, value_off, 0)
    return builder.EndObject()


def _build_backend_delegate(builder: Any, val: BackendDelegate) -> int:
    id_off = builder.CreateString(val.id)
    processed_off = _build_backend_delegate_data_reference(builder, val.processed)
    compile_spec_offsets = [
        _build_compile_spec(builder, spec) for spec in val.compile_specs
    ]
    compile_specs_off = _create_uoffset_vector(builder, compile_spec_offsets)
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(0, id_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, processed_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, compile_specs_off, 0)
    return builder.EndObject()


def _build_chain(
    builder: Any,
    val: Chain,
    *,
    instr_union_ids: Dict[str, int],
) -> int:
    inputs_off = _create_int32_vector(builder, val.inputs)
    outputs_off = _create_int32_vector(builder, val.outputs)
    instr_offsets = [
        _build_instruction(builder, instr, instr_union_ids=instr_union_ids)
        for instr in val.instructions
    ]
    instructions_off = _create_uoffset_vector(builder, instr_offsets)
    stacktrace_off = (
        _create_uoffset_vector(
            builder,
            [_build_frame_list(builder, fl) for fl in val.stacktrace],
        )
        if val.stacktrace is not None
        else 0
    )

    builder.StartObject(4)
    builder.PrependUOffsetTRelativeSlot(0, inputs_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, outputs_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, instructions_off, 0)
    if stacktrace_off != 0:
        builder.PrependUOffsetTRelativeSlot(3, stacktrace_off, 0)
    return builder.EndObject()


def _build_execution_plan(
    builder: Any,
    val: ExecutionPlan,
    *,
    kernel_union_ids: Dict[str, int],
    instr_union_ids: Dict[str, int],
) -> int:
    name_off = builder.CreateString(val.name)
    container_meta_off = _build_container_metadata(builder, val.container_meta_type)
    evalue_offsets = [
        _build_evalue(builder, evalue, kernel_union_ids=kernel_union_ids)
        for evalue in val.values
    ]
    values_off = _create_uoffset_vector(builder, evalue_offsets)
    inputs_off = _create_int32_vector(builder, val.inputs)
    outputs_off = _create_int32_vector(builder, val.outputs)
    chain_offsets = [
        _build_chain(builder, chain, instr_union_ids=instr_union_ids)
        for chain in val.chains
    ]
    chains_off = _create_uoffset_vector(builder, chain_offsets)
    operator_offsets = [_build_operator(builder, op) for op in val.operators]
    operators_off = _create_uoffset_vector(builder, operator_offsets)
    delegate_offsets = [_build_backend_delegate(builder, d) for d in val.delegates]
    delegates_off = _create_uoffset_vector(builder, delegate_offsets)
    non_const_buf_sizes_off = _create_int64_vector(builder, val.non_const_buffer_sizes)

    builder.StartObject(9)
    builder.PrependUOffsetTRelativeSlot(0, name_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, container_meta_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, values_off, 0)
    builder.PrependUOffsetTRelativeSlot(3, inputs_off, 0)
    builder.PrependUOffsetTRelativeSlot(4, outputs_off, 0)
    builder.PrependUOffsetTRelativeSlot(5, chains_off, 0)
    builder.PrependUOffsetTRelativeSlot(6, operators_off, 0)
    builder.PrependUOffsetTRelativeSlot(7, delegates_off, 0)
    builder.PrependUOffsetTRelativeSlot(8, non_const_buf_sizes_off, 0)
    return builder.EndObject()


def _build_buffer(builder: Any, val: Buffer, *, tensor_alignment: int) -> int:
    storage_off = _create_aligned_byte_vector(builder, val.storage, tensor_alignment)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, storage_off, 0)
    return builder.EndObject()


def _build_backend_delegate_inline_data(
    builder: Any, val: BackendDelegateInlineData, *, delegate_alignment: int
) -> int:
    data_off = _create_aligned_byte_vector(builder, val.data, delegate_alignment)
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, data_off, 0)
    return builder.EndObject()


def _build_data_segment(builder: Any, val: DataSegment) -> int:
    builder.StartObject(2)
    builder.PrependUint64Slot(0, int(val.offset), 0)
    builder.PrependUint64Slot(1, int(val.size), 0)
    return builder.EndObject()


def _build_subsegment_offsets(builder: Any, val: SubsegmentOffsets) -> int:
    offsets_off = _create_uint64_vector(builder, val.offsets)
    builder.StartObject(2)
    builder.PrependUint32Slot(0, int(val.segment_index), 0)
    builder.PrependUOffsetTRelativeSlot(1, offsets_off, 0)
    return builder.EndObject()


def _build_named_data(builder: Any, val: NamedData) -> int:
    key_off = builder.CreateString(val.key)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, key_off, 0)
    builder.PrependUint32Slot(1, int(val.segment_index), 0)
    return builder.EndObject()


def _program_to_flatbuffer(
    program: Program,
    *,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> _FlatbufferResult:
    """Converts a Program dataclass into binary flatbuffer data.

    Unlike _program_json_to_flatbuffer(), this does not use JSON or invoke
    flatc to build the binary.
    """
    file_identifier, tensor_alignment, delegate_alignment_eff, max_alignment = (
        _get_schema_details(
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
    )
    kernel_union_ids, instr_union_ids = _get_union_type_ids()

    builder: Any = flatbuffers.Builder(0)

    exec_plan_offsets = [
        _build_execution_plan(
            builder,
            plan,
            kernel_union_ids=kernel_union_ids,
            instr_union_ids=instr_union_ids,
        )
        for plan in program.execution_plan
    ]
    execution_plan_off = _create_uoffset_vector(builder, exec_plan_offsets)

    constant_buffer_offsets = [
        _build_buffer(builder, buf, tensor_alignment=tensor_alignment)
        for buf in program.constant_buffer
    ]
    constant_buffer_off = _create_uoffset_vector(builder, constant_buffer_offsets)

    backend_delegate_data_offsets = [
        _build_backend_delegate_inline_data(
            builder, d, delegate_alignment=delegate_alignment_eff
        )
        for d in program.backend_delegate_data
    ]
    backend_delegate_data_off = _create_uoffset_vector(
        builder, backend_delegate_data_offsets
    )

    segment_offsets = [_build_data_segment(builder, s) for s in program.segments]
    segments_off = _create_uoffset_vector(builder, segment_offsets)

    constant_segment_off = _build_subsegment_offsets(builder, program.constant_segment)

    mutable_data_segments_off = (
        _create_uoffset_vector(
            builder,
            [
                _build_subsegment_offsets(builder, s)
                for s in program.mutable_data_segments
            ],
        )
        if program.mutable_data_segments is not None
        else 0
    )
    named_data_off = (
        _create_uoffset_vector(
            builder, [_build_named_data(builder, nd) for nd in program.named_data]
        )
        if program.named_data is not None
        else 0
    )

    builder.StartObject(8)
    builder.PrependUint32Slot(0, int(program.version), 0)
    builder.PrependUOffsetTRelativeSlot(1, execution_plan_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, constant_buffer_off, 0)
    builder.PrependUOffsetTRelativeSlot(3, backend_delegate_data_off, 0)
    builder.PrependUOffsetTRelativeSlot(4, segments_off, 0)
    builder.PrependUOffsetTRelativeSlot(5, constant_segment_off, 0)
    if mutable_data_segments_off != 0:
        builder.PrependUOffsetTRelativeSlot(6, mutable_data_segments_off, 0)
    if named_data_off != 0:
        builder.PrependUOffsetTRelativeSlot(7, named_data_off, 0)
    program_off = builder.EndObject()

    builder.Finish(program_off, file_identifier)
    return _FlatbufferResult(data=bytes(builder.Output()), max_alignment=max_alignment)
