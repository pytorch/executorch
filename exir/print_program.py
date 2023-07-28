# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import reprlib
from dataclasses import fields
from enum import IntEnum
from typing import Any, List

import torch

from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.schema import (
    Bool,
    BoolList,
    DelegateCall,
    Double,
    DoubleList,
    EValue,
    FrameList,
    FreeCall,
    Int,
    IntList,
    JumpFalseCall,
    KernelCall,
    MoveCall,
    Null,
    OptionalTensorList,
    Program,
    ScalarType,
    String,
    Tensor,
    TensorList,
    TensorShapeDynamism,
)


def _scalar_type_str(scalar_type: ScalarType) -> str:
    type2str = {
        ScalarType.BYTE: "bt",
        ScalarType.CHAR: "c",
        ScalarType.SHORT: "s",
        ScalarType.INT: "i",
        ScalarType.LONG: "l",
        ScalarType.HALF: "h",
        ScalarType.FLOAT: "f",
        ScalarType.DOUBLE: "d",
        ScalarType.COMPLEX32: "c32",
        ScalarType.COMPLEX64: "c64",
        ScalarType.COMPLEX128: "c128",
        ScalarType.BOOL: "b",
        ScalarType.QINT8: "qi8",
        ScalarType.QUINT8: "qui8",
        ScalarType.QINT32: "qi32",
        ScalarType.BFLOAT16: "bf16",
        ScalarType.QUINT4x2: "qui4x2",
        ScalarType.QUINT2x4: "qui2x4",
    }
    if not (ret := type2str.get(scalar_type, None)):
        raise RuntimeError(f"Unrecognized scalar_type: {scalar_type}")
    else:
        return ret


def _is_dynamic_shape_tensor(tensor: Tensor) -> bool:
    return tensor.shape_dynamism != TensorShapeDynamism.STATIC


def _format_evalue(
    evalue: EValue, show_meminfo: bool, mark_dynamic_shape_tensor: bool
) -> str:
    evstr = "\033[34m"
    if isinstance(evalue.val, Tensor):
        tensor = evalue.val
        if tensor.constant_buffer_idx > 0:
            assert not _is_dynamic_shape_tensor(
                tensor
            ), "A constant tensor can not be dynamic shape"
            evstr += "CT"  # constant tensor
            assert tensor.allocation_info is None
        else:
            if mark_dynamic_shape_tensor:
                if tensor.shape_dynamism == TensorShapeDynamism.DYNAMIC_BOUND:
                    evstr += "UB"  # upper bound tensor will be shown as 'UBT'
                elif tensor.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND:
                    evstr += "DU"  # dynamic unbound tensor will be shown as 'DUT'
            evstr += "T"
            if show_meminfo:
                if tensor.allocation_info:
                    evstr += f"m{tensor.allocation_info.memory_id}.{tensor.allocation_info.memory_offset}"
                else:
                    evstr += "m."
        evstr += f"{tensor.sizes}{_scalar_type_str(tensor.scalar_type)}"
    elif isinstance(evalue.val, TensorList):
        evstr += "TL"
        tensorlist = evalue.val
        # pyre-ignore
        evstr += str(tensorlist.items)
    elif isinstance(evalue.val, OptionalTensorList):
        evstr += "OTL"
        optionaltensorlist = evalue.val
        # pyre-ignore
        evstr += str(optionaltensorlist.items)
    elif isinstance(evalue.val, IntList):
        evstr += "IL"
        intlist = evalue.val
        # pyre-ignore
        evstr += str(intlist.items)
    elif isinstance(evalue.val, DoubleList):
        evstr += "DL"
        doublelist = evalue.val
        # pyre-ignore
        evstr += str(doublelist.items)
    elif isinstance(evalue.val, BoolList):
        evstr += "BL"
        boollist = evalue.val
        # pyre-ignore
        evstr += str(boollist.items)
    elif isinstance(evalue.val, Int):
        intval = evalue.val
        evstr += f"I{intval.int_val}"
    elif isinstance(evalue.val, Double):
        doubleval = evalue.val
        evstr += f"D{doubleval.double_val}"
    elif isinstance(evalue.val, Bool):
        boolval = evalue.val
        evstr += f"B{int(boolval.bool_val)}"  # print 0, 1 since it's shorter than false, true
    elif isinstance(evalue.val, String):
        stringval = evalue.val
        evstr += f"S{stringval.string_val}"
    elif isinstance(evalue.val, Null):
        evstr += "N"  # for null
    else:
        raise RuntimeError(f"Unrecognized type of evalue: {evalue}")
    evstr += "\033[0m"
    return evstr


def print_program(
    program: Program, show_meminfo: bool = True, mark_dynamic_shape_tensor: bool = False
) -> None:
    """
    Dump the instruction list of a program in a more human readable fashion.

    The dump follows the following BNF syntax (I combime some regex syntax
    so the grammar becomes shorter. The grammar is not strict but the main
    purpose is to let people understand the dump):
    ```
      PROGRAM: (INSTRUCTION)+
      INSTRUCTION: SEQUENCE_NO ':' (CALL_KERNEL | JUMP_FALSE)
      JUMP_FALSE: 'JF' '(' EVALUE ')' '->' TARGET_SEQUENCE_NO
      CALL_KERNEL: OVERLOADDED_OP_NAME ARGS
      ARGS: EVALUE | ARGS ',' EVALUE
      EVALUE: EVALUE_IDX ( TENSOR | INT | BOOL | ...)
      INT: 'I' ACTUAL_INT_VALUE
      BOOL: 'B' ZERO_OR_ONE
      CONST_TENSOR_PREFIX: 'CT'
      TENSOR: ('T' | CONST_TENSOR_PREFIX) (MEM_ALLOCATION_INFO)? TENSOR_SHAPE TENSOR_DTYPE
      TENSOR_SHAPE: '[' dim0_size, dim1_size, ..., last_dim_size ']'
      MEM_ALLOCATION_INFO: PLANNED_MEM_INFO | UNPLANNED_MEM_INFO
      PLANNED_MEM_INFO: 'm' MEM_LAYER_ID '.' MEM_LAYER_OFFSET
      UNPLANNED_MEM_INFO: 'm.'
    ```

    To make the dump easier to read, it's colored as follows:
    1. input/output EValues are marked as red
    2. EValue types (or more specifically tensor types with size and dtype) are marked as blue
    """
    execution_plan = program.execution_plan[0]
    operators = execution_plan.operators
    delegates = execution_plan.delegates
    chain = execution_plan.chains[0]
    instructions = chain.instructions
    inputs: List[int] = execution_plan.inputs
    outputs: List[int] = execution_plan.outputs
    values: List[EValue] = execution_plan.values

    def _format_arg(evalue_idx: int) -> str:
        def _get_io_index(iolist: List[int], target_evalue_idx: int) -> int:
            """
            The list is short enough so linear scan is proper.
            """
            for io_idx, evalue_idx in enumerate(iolist):
                if evalue_idx == target_evalue_idx:
                    return io_idx
            return -1

        argstr = str(evalue_idx)
        if (input_idx := _get_io_index(inputs, evalue_idx)) >= 0:
            argstr += f"\033[31mI{input_idx}\033[0m"
        if (output_idx := _get_io_index(outputs, evalue_idx)) >= 0:
            argstr += f"\033[31mO{output_idx}\033[0m"

        # EValue type
        evalue = values[evalue_idx]
        return argstr + _format_evalue(evalue, show_meminfo, mark_dynamic_shape_tensor)

    print(f"The program contains the following {len(instructions)} instructions")
    for idx, instr in enumerate(instructions):
        print(f"{idx:3}: ", end="")
        if isinstance(instr.instr_args, KernelCall):
            kernel = instr.instr_args
            op = operators[kernel.op_index]
            args = kernel.args

            opname = f"{op.name}.{op.overload}" if op.overload else op.name
            argstr = ",".join(map(_format_arg, args))
            print(f"{opname} {argstr}")
        elif isinstance(instr.instr_args, DelegateCall):
            delegate = instr.instr_args
            backend = delegates[delegate.delegate_index]
            args = delegate.args
            backend_id = f"{backend.id}"
            argstr = ",".join(map(_format_arg, args))
            print(f"{backend_id} {argstr}")
        elif isinstance(instr.instr_args, JumpFalseCall):
            jfcall = instr.instr_args
            print(
                f"JF ({_format_arg(jfcall.cond_value_index)}) -> {jfcall.destination_instruction}"
            )
        elif isinstance(instr.instr_args, MoveCall):
            move_call = instr.instr_args
            print(
                f"MOVE {_format_arg(move_call.move_from)} -> {_format_arg(move_call.move_to)}"
            )
        elif isinstance(instr.instr_args, FreeCall):
            print(f"FREE {_format_arg(instr.instr_args.value_index)}")
        else:
            raise InternalError(f"Unsupport instruction type {instr}")


# pyre-ignore
def pretty_print(obj: Any, indent: int = 0) -> None:
    """
    Pretty prints the given object which is of the Program type and any of its
    attribute’s types.
    """
    if isinstance(obj, torch.fx.GraphModule):
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            "pretty_print() does not accept GraphModule as input.",
        )

    # Instruction types are IntEnum object
    if isinstance(obj, IntEnum):
        print(int(obj), end="")
        return

    primitives = (int, str, bool, float, type(None))
    if isinstance(obj, primitives):
        print(obj, end="")
        return

    if isinstance(obj, bytes):
        r = reprlib.Repr()
        r.maxother = 1024
        print(r.repr(obj), end="")
        return

    if isinstance(obj, list):
        if len(obj) < 10 and all(isinstance(elem, int) for elem in obj):
            print(obj, end="")
            return
        print("[")
        for index, elem in enumerate(obj):
            print("  " * (indent + 1), end="")
            pretty_print(elem, indent + 1)
            print(f"(index={index}),")
        print("  " * indent + "]", end="")
        return

    inline = all(
        isinstance(getattr(obj, field.name), primitives) for field in fields(obj)
    )
    end = "" if inline else "\n"
    print(f"{type(obj).__name__}(", end=end)
    for i, _field in enumerate(fields(obj)):
        if not inline:
            print("  " * (indent + 1), end="")
        print(_field.name + "=", end="")
        pretty_print(getattr(obj, _field.name), indent + 1)
        if i < len(fields(obj)) - 1:
            print(", ", end="")
        print("", end=end)
    if not inline:
        print("  " * indent, end="")
    print(")", end="" if indent else "\n")


def pretty_print_stacktraces(obj: FrameList) -> str:
    """
    Pretty prints the traceback for one instruction
    """
    pretty = "Traceback (most recent call last): \n"
    for frame in obj.items:
        pretty += f'    File "{frame.filename}", '
        pretty += f"line {str(frame.lineno)}, in {frame.name}\n"
        pretty += f"{frame.context} \n"
    pretty += "\n"
    return pretty
