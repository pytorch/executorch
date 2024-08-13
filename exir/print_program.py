# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import re
import reprlib
from dataclasses import fields
from enum import IntEnum
from typing import Any, List, Optional, TextIO

import torch
from executorch.exir.error import ExportError, ExportErrorType, InternalError

from executorch.exir.schema import (
    Bool,
    BoolList,
    DelegateCall,
    Double,
    DoubleList,
    EValue,
    Frame,
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


def _format_evalue(  # noqa: C901
    evalue: EValue, show_meminfo: bool, mark_dynamic_shape_tensor: bool
) -> str:
    evstr = "\033[34m"
    if isinstance(evalue.val, Tensor):
        tensor = evalue.val
        if tensor.data_buffer_idx > 0:
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


def print_program(  # noqa: C901
    program: Program,
    show_meminfo: bool = True,
    mark_dynamic_shape_tensor: bool = False,
    out: Optional[TextIO] = None,
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

    print(
        f"The program contains the following {len(instructions)} instructions", file=out
    )
    for idx, instr in enumerate(instructions):
        print(f"{idx:3}: ", end="", file=out)
        if isinstance(instr.instr_args, KernelCall):
            kernel = instr.instr_args
            op = operators[kernel.op_index]
            args = kernel.args

            opname = f"{op.name}.{op.overload}" if op.overload else op.name
            argstr = ",".join(map(_format_arg, args))
            print(f"{opname} {argstr}", file=out)
        elif isinstance(instr.instr_args, DelegateCall):
            delegate = instr.instr_args
            backend = delegates[delegate.delegate_index]
            args = delegate.args
            backend_id = f"{backend.id}"
            argstr = ",".join(map(_format_arg, args))
            print(f"{backend_id} {argstr}", file=out)
        elif isinstance(instr.instr_args, JumpFalseCall):
            jfcall = instr.instr_args
            print(
                f"JF ({_format_arg(jfcall.cond_value_index)}) -> {jfcall.destination_instruction}",
                file=out,
            )
        elif isinstance(instr.instr_args, MoveCall):
            move_call = instr.instr_args
            print(
                f"MOVE {_format_arg(move_call.move_from)} -> {_format_arg(move_call.move_to)}",
                file=out,
            )
        elif isinstance(instr.instr_args, FreeCall):
            print(f"FREE {_format_arg(instr.instr_args.value_index)}", file=out)
        else:
            raise InternalError(f"Unsupport instruction type {instr}")


# pyre-ignore
def pretty_print(obj: Any, indent: int = 0, out: Optional[TextIO] = None) -> None:
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
        print(int(obj), end="", file=out)
        return

    primitives = (int, str, bool, float, type(None))
    if isinstance(obj, primitives):
        print(obj, end="", file=out)
        return

    if isinstance(obj, bytes):
        r = reprlib.Repr()
        r.maxother = 1024
        print(r.repr(obj), end="", file=out)
        return

    if isinstance(obj, list):
        if len(obj) < 10 and all(isinstance(elem, int) for elem in obj):
            print(obj, end="", file=out)
            return
        print("[", file=out)
        for index, elem in enumerate(obj):
            print("  " * (indent + 1), end="", file=out)
            pretty_print(elem, indent + 1, out=out)
            print(f"(index={index}),", file=out)
        print("  " * indent + "]", end="", file=out)
        return

    inline = all(
        isinstance(getattr(obj, field.name), primitives) for field in fields(obj)
    )
    end = "" if inline else "\n"
    print(f"{type(obj).__name__}(", end=end, file=out)
    for i, _field in enumerate(fields(obj)):
        if not inline:
            print("  " * (indent + 1), end="", file=out)
        print(_field.name + "=", end="", file=out)
        pretty_print(getattr(obj, _field.name), indent + 1, out=out)
        if i < len(fields(obj)) - 1:
            print(", ", end="", file=out)
        print("", end=end, file=out)
    if not inline:
        print("  " * indent, end="", file=out)
    print(")", end="" if indent else "\n", file=out)


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


def add_cursor_to_graph(graph: torch.fx.Graph, finding_node: torch.fx.Node) -> str:
    """
    Insert a cursor at the node location in the fx.Graph.
    e.g:
    # graph():
    #   %x : [#users=1] = placeholder[target=x]
    #   %param : [#users=1] = get_attr[target=param]
    #   %add : [#users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    # --> %linear : [#users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    #   %clamp : [#users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    #   return clamp

    This is mostly used for error reporting
    """

    new_graph = copy.deepcopy(graph)

    found_at = -1
    for ix, node in enumerate(graph.nodes):
        if node == finding_node:
            found_at = ix

    # This is heavily based on __str__ method of fx.Graph
    def _format_graph(graph: torch.fx.Graph, offending_node_idx: int) -> str:
        s = "graph():"
        for ix, node in enumerate(graph.nodes):
            node_str = node.format_node()
            if node_str:
                if ix != offending_node_idx:
                    s += "\n    " + node_str
                else:
                    s += "\n--> " + node_str
        return s

    return _format_graph(new_graph, found_at)


def _stacktrace_to_framelist(stacktrace: str) -> FrameList:
    """Creates a frame list from a stacktrace string."""
    pattern = r'File "(.*?)", line (\d+), in (.*?)\n'
    matches = re.findall(pattern, stacktrace)
    mapped_frame_list = [
        Frame(
            filename=match[0],
            lineno=int(match[1]),
            name=match[2],
            context=stacktrace.split("\n")[i * 2 + 1].strip(),
        )
        for i, match in enumerate(matches)
    ]
    return FrameList(mapped_frame_list)


def inspect_node(graph: torch.fx.Graph, node: torch.fx.Node) -> str:
    """
    Inspect a node by highlighting the node in the graph as well as the stacktrace.

    Args:
        graph: The graph containing the node
        node: The node to be inspected

    Return: A string. An example output is:

    _param_constant0 error_msg:  Here is the failing node in the graph module:
    graph():
        %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    --> %_param_constant0 : [num_users=1] = get_attr[target=_param_constant0]
        %_param_constant1 : [num_users=1] = get_attr[target=_param_constant1]
        %aten_convolution_default : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%arg0_1, %_param_constant0, %_param_constant1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
        %_param_constant2 : [num_users=1] = get_attr[target=_param_constant2]
        %_param_constant3 : [num_users=1] = get_attr[target=_param_constant3]
        %aten_convolution_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_convolution_default, %_param_constant2, %_param_constant3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
        %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_convolution_default, %aten_convolution_default_1), kwargs = {})
        %_param_constant4 : [num_users=1] = get_attr[target=_param_constant4]
        %_param_constant5 : [num_users=1] = get_attr[target=_param_constant5]
        %aten_convolution_default_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor, %_param_constant4, %_param_constant5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
        %aten_gelu_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.gelu.default](args = (%aten_convolution_default_2,), kwargs = {})
        return [aten_gelu_default]
    This node _param_constant0 has metadata of:
    The node stacktrace:
    Traceback (most recent call last):
        File "/tmp/ipykernel_1204253/3382880687.py", line 7, in forward
    return self.test_model(x)
        File "/mnt/xarfuse/uid-25337/7b86ad0c-seed-nspid4026532987_cgpid2707357-ns-4026532984/torch/nn/modules/module.py", line 1528, in _call_impl
    return forward_call(*args, **kwargs)
        File "/tmp/ipykernel_1204253/712280972.py", line 10, in forward
    a = self.conv1(x)

    """
    graph_str_with_cursor = add_cursor_to_graph(graph, node)
    error_msg = (
        f"Here is the node in the graph module:\n"
        f"{graph_str_with_cursor}\n"
        f"This node {node} has metadata of:\n"
    )
    # Node spec error message
    if hasattr(node.meta, "spec"):
        error_msg += f"The node spec:\n{node.meta['spec']}\n"

    # Stacktrace error message
    if "stack_trace" in node.meta:
        framelist = _stacktrace_to_framelist(node.meta["stack_trace"])
        error_msg += f"The node stacktrace:\n{pretty_print_stacktraces(framelist)}\n"
    return error_msg
