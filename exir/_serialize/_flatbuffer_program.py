# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
import functools
import importlib
import pkgutil
import tempfile

from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Dict, get_args, get_origin, get_type_hints, Optional, Union

import flatbuffers  # pyre-ignore[21]
from executorch.exir._serialize._flatbuffer import (
    _FlatbufferResult,
    _is_valid_alignment,
    _prepare_schema,
    _SchemaInfo,
)
from executorch.exir._serialize.generated import executorch_flatbuffer as _generated_fb
from executorch.exir._serialize.generated.executorch_flatbuffer import (
    BackendDelegateInlineData as _BackendDelegateInlineData,
    Buffer as _Buffer,
    InstructionArguments as _InstructionArguments,
    KernelTypes as _KernelTypes,
)
from executorch.exir._serialize.generated.executorch_flatbuffer.Program import ProgramT
from executorch.exir.schema import Double, EValue, Instruction, Program

_T_CLASS_CACHE: Dict[type, type] = {}
_FIELD_NAME_CACHE: Dict[type, tuple[tuple[str, str], ...]] = {}
_TYPE_HINTS_CACHE: Dict[type, Dict[str, Any]] = {}
_BUFFER_ALIGNMENT: ContextVar[int] = ContextVar("_BUFFER_ALIGNMENT", default=1)
_DELEGATE_ALIGNMENT: ContextVar[int] = ContextVar("_DELEGATE_ALIGNMENT", default=1)


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _flatbuffer_t_class(dataclass_type: type) -> type:
    cached = _T_CLASS_CACHE.get(dataclass_type)
    if cached is not None:
        return cached
    module_name = f"executorch.exir._serialize.generated.executorch_flatbuffer.{dataclass_type.__name__}"
    module = importlib.import_module(module_name)
    t_cls = getattr(module, f"{dataclass_type.__name__}T")
    _T_CLASS_CACHE[dataclass_type] = t_cls
    return t_cls


def _dataclass_field_map(dataclass_type: type) -> tuple[tuple[str, str], ...]:
    cached = _FIELD_NAME_CACHE.get(dataclass_type)
    if cached is not None:
        return cached
    mapping = tuple(
        (field.name, _snake_to_camel(field.name)) for field in fields(dataclass_type)
    )
    _FIELD_NAME_CACHE[dataclass_type] = mapping
    return mapping


def _dataclass_type_hints(dataclass_type: type) -> Dict[str, Any]:
    cached = _TYPE_HINTS_CACHE.get(dataclass_type)
    if cached is not None:
        return cached
    type_hints = get_type_hints(dataclass_type)
    _TYPE_HINTS_CACHE[dataclass_type] = type_hints
    return type_hints


def _create_aligned_byte_vector(builder: Any, data: bytes, alignment: int) -> int:
    if not _is_valid_alignment(alignment):
        raise ValueError(f"Bad alignment {alignment}")
    builder.StartVector(1, len(data), alignment)
    length = len(data)
    builder.head = builder.Head() - length  # pyre-ignore[16]
    builder.Bytes[builder.Head() : builder.Head() + length] = data  # pyre-ignore[16]
    return builder.EndVector()


def _coerce_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    tobytes = getattr(data, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    return bytes(data)


def _pack_buffer(self: Any, builder: Any) -> int:
    storage = 0
    if self.storage is not None:
        storage = _create_aligned_byte_vector(
            builder, _coerce_bytes(self.storage), _BUFFER_ALIGNMENT.get()
        )
    _Buffer.BufferStart(builder)
    if storage:
        _Buffer.BufferAddStorage(builder, storage)
    return _Buffer.BufferEnd(builder)


def _pack_backend_delegate_inline_data(self: Any, builder: Any) -> int:
    data = 0
    if self.data is not None:
        data = _create_aligned_byte_vector(
            builder, _coerce_bytes(self.data), _DELEGATE_ALIGNMENT.get()
        )
    _BackendDelegateInlineData.BackendDelegateInlineDataStart(builder)
    if data:
        _BackendDelegateInlineData.BackendDelegateInlineDataAddData(builder, data)
    return _BackendDelegateInlineData.BackendDelegateInlineDataEnd(builder)


@functools.lru_cache(maxsize=1)
def _install_fast_packers() -> None:
    _Buffer.BufferT.Pack = _pack_buffer
    _BackendDelegateInlineData.BackendDelegateInlineDataT.Pack = (
        _pack_backend_delegate_inline_data
    )


def _set_pack_alignments(tensor_alignment: int, delegate_alignment: int) -> None:
    _BUFFER_ALIGNMENT.set(tensor_alignment)
    _DELEGATE_ALIGNMENT.set(delegate_alignment)


def _convert_double(val: Double) -> Any:
    result = _flatbuffer_t_class(Double)()
    double_val = val.double_val
    if isinstance(double_val, str):
        # the string is read as a Union of float and string (see exir/schema.py).
        result.doubleVal = float(double_val)
    else:
        result.doubleVal = double_val
    return result


def _convert_evalue(val: EValue) -> Any:
    result = _flatbuffer_t_class(EValue)()
    union_val = val.val
    if union_val is None:
        result.valType = _KernelTypes.KernelTypes.NONE
        result.val = None
        return result
    union_name = type(union_val).__name__
    result.valType = getattr(_KernelTypes.KernelTypes, union_name)
    result.val = _convert_value(union_val)
    return result


def _convert_instruction(val: Instruction) -> Any:
    result = _flatbuffer_t_class(Instruction)()
    union_val = val.instr_args
    if union_val is None:
        result.instrArgsType = _InstructionArguments.InstructionArguments.NONE
        result.instrArgs = None
        return result
    union_name = type(union_val).__name__
    result.instrArgsType = getattr(
        _InstructionArguments.InstructionArguments, union_name
    )
    result.instrArgs = _convert_value(union_val)
    return result


def _convert_dataclass(val: Any) -> Any:
    result = _flatbuffer_t_class(type(val))()
    for src_name, dst_name in _dataclass_field_map(type(val)):
        setattr(result, dst_name, _convert_value(getattr(val, src_name)))
    return result


def _convert_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, enum.Enum):
        return int(val)
    if isinstance(val, Double):
        return _convert_double(val)
    if isinstance(val, EValue):
        return _convert_evalue(val)
    if isinstance(val, Instruction):
        return _convert_instruction(val)
    if isinstance(val, (bytes, bytearray)):
        return val
    if isinstance(val, (list, tuple)):
        return [_convert_value(item) for item in val]
    if is_dataclass(val):
        return _convert_dataclass(val)
    return val


def convert_program(val: Program) -> ProgramT:
    return _convert_dataclass(val)


# The generated FlatBuffer Python modules import child tables/unions as modules
# (for example, Program.ExecutionPlan becomes the ExecutionPlan module), but the
# unpacking helpers later expect those globals to be the corresponding classes.
# Rebind module globals like ExecutionPlan -> ExecutionPlan.ExecutionPlan so the
# generated InitFromObj()/InitFromPackedBuf() code can instantiate nested types.
def _patch_generated_module_aliases(module: ModuleType) -> None:
    for name, maybe_module in vars(module).items():
        if not isinstance(maybe_module, ModuleType):
            continue
        maybe_class = getattr(maybe_module, name, None)
        if isinstance(maybe_class, type):
            setattr(module, name, maybe_class)


@lru_cache(maxsize=1)
def _patch_generated_flatbuffer_aliases() -> None:
    package_name = _generated_fb.__name__
    for module_info in pkgutil.iter_modules(_generated_fb.__path__):
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        _patch_generated_module_aliases(module)


def _flatbuffer_dataclass_names(val: Any) -> tuple[str, Optional[str]]:
    val_type_name = type(val).__name__
    if val_type_name.endswith("T"):
        return val_type_name, val_type_name[:-1]
    return val_type_name, None


def _matches_dataclass_union_type(
    union_type: Any, val_type_name: str, val_dataclass_name: Optional[str]
) -> bool:
    if not is_dataclass(union_type):
        return False
    union_name = union_type.__name__
    return union_name == val_type_name or (
        val_dataclass_name is not None and union_name == val_dataclass_name
    )


def _matches_non_dataclass_union_type(union_type: Any, val: Any) -> bool:
    if union_type is Any:
        return True
    if union_type is str and isinstance(val, (bytes, bytearray, memoryview)):
        return True
    union_origin = get_origin(union_type)
    if union_origin is list and hasattr(val, "__iter__"):
        return True
    return isinstance(union_type, type) and isinstance(val, union_type)


def _union_choice_from_value(union_types: tuple[Any, ...], val: Any) -> Any:
    if val is None:
        for union_type in union_types:
            if union_type is type(None):
                return union_type
        return None

    val_type_name, val_dataclass_name = _flatbuffer_dataclass_names(val)

    for union_type in union_types:
        if union_type is type(None):
            continue
        if _matches_dataclass_union_type(union_type, val_type_name, val_dataclass_name):
            return union_type
        if _matches_non_dataclass_union_type(union_type, val):
            return union_type
    return None


def _convert_from_flatbuffer_value(val: Any, expected_type: Any) -> Any:
    if val is None:
        return None

    origin = get_origin(expected_type)
    if origin is list:
        item_type = get_args(expected_type)[0]
        return [_convert_from_flatbuffer_value(item, item_type) for item in val]

    if origin is Union:
        union_type = _union_choice_from_value(get_args(expected_type), val)
        if union_type is None:
            raise TypeError(
                f"Could not match value type {type(val)} to {expected_type}"
            )
        if union_type is type(None):
            return None
        return _convert_from_flatbuffer_value(val, union_type)

    if expected_type is bytes:
        return _coerce_bytes(val)
    if expected_type is str and isinstance(val, (bytes, bytearray, memoryview)):
        return _coerce_bytes(val).decode("utf-8")
    if is_dataclass(expected_type):
        return _convert_from_flatbuffer_dataclass(val, expected_type)
    if isinstance(expected_type, type) and issubclass(expected_type, enum.Enum):
        if isinstance(val, expected_type):
            return val
        return expected_type(val)
    if isinstance(expected_type, type):
        return expected_type(val)
    return val


def _convert_from_flatbuffer_dataclass(val: Any, dataclass_type: type) -> Any:
    result = {}
    type_hints = _dataclass_type_hints(dataclass_type)
    for src_name, dst_name in _dataclass_field_map(dataclass_type):
        result[src_name] = _convert_from_flatbuffer_value(
            getattr(val, dst_name), type_hints[src_name]
        )
    return dataclass_type(**result)


def _flatbuffer_to_program(program_data: bytes) -> Program:
    _patch_generated_flatbuffer_aliases()
    program_t = ProgramT.InitFromPackedBuf(program_data)
    return _convert_from_flatbuffer_dataclass(program_t, Program)


@lru_cache(maxsize=1)
def _get_schema_info(
    constant_tensor_alignment: Optional[int], delegate_alignment: Optional[int]
) -> _SchemaInfo:
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_info = _prepare_schema(
            out_dir=temp_dir,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
    return schema_info


def _program_to_flatbuffer(
    program: Program,
    *,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> _FlatbufferResult:
    """Converts a Program dataclass into binary flatbuffer data."""
    schema_info = _get_schema_info(constant_tensor_alignment, delegate_alignment)
    _set_pack_alignments(schema_info.tensor_alignment, schema_info.delegate_alignment)
    _install_fast_packers()
    program_t = convert_program(program)
    builder = flatbuffers.Builder()
    program_offset = program_t.Pack(builder)
    builder.Finish(program_offset, file_identifier=schema_info.file_identifier)
    return _FlatbufferResult(
        data=bytes(builder.Output()), max_alignment=schema_info.max_alignment
    )
