# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import List, Optional, Union

# pyre-fixme[21]: Could not find module `executorch.exir.verification.bindings`.
import executorch.exir.verification.bindings as bindings  # @manual=//executorch/exir/verification:bindings
import executorch.extension.pytree as ex_pytree

import torch

from executorch import exir

from executorch.exir.schema import (
    Bool,
    BoolList,
    Double,
    DoubleList,
    ExecutionPlan,
    Int,
    IntList,
    JumpFalseCall,
    KernelCall,
    KernelTypes,
    MoveCall,
    Null,
    Operator,
    OptionalTensorList,
    Program,
    String,
    Tensor,
    TensorList,
)
from executorch.exir.tensor import get_scalar_type, stride_from_dim_order
from torch.library import impl, Library
from torch.utils._pytree import PyTree


class Uninitialized:
    pass


ValueListType = Union[
    List[torch.Tensor],
    List[bool],
    List[float],
    List[int],
]

ValueScalarType = Union[int, str, float, torch.Tensor, Uninitialized, None]

ValueType = Union[
    ValueScalarType,
    ValueListType,
]

# defining the operator executorch.move
executorch_lib = Library("executorch", "DEF")
executorch_lib.define("move(Tensor self) -> Tensor")


@impl(executorch_lib, "move", "CPU")
def move_impl(self: torch.Tensor) -> torch.Tensor:
    return torch.clone(self)


def comp_types(val: KernelTypes, input_val: ValueType) -> Optional[bool]:  # noqa
    """
    Compares a schema type (val) with Python type (input_val)
    Map: Int -> int, Bool -> bool, Double -> float,
         String -> str, Tensor -> torch.Tensor, XList -> list[x]

    Args:
    `val`: value from value list with type from `schema.py`

    `input_val`: value with Python type, normally from the result of executing an operation
    """
    if isinstance(val, Int):
        return isinstance(input_val, type(val.int_val))
    elif isinstance(val, Bool):
        return isinstance(input_val, type(val.bool_val))
    elif isinstance(val, Double):
        return isinstance(input_val, type(val.double_val))
    elif isinstance(val, String):
        return isinstance(input_val, type(val.string_val))
    elif isinstance(val, Tensor):
        return isinstance(input_val, torch.Tensor)
    elif isinstance(val, IntList):
        if not isinstance(input_val, list):
            return False
        return all(isinstance(x, int) for x in input_val)
    elif isinstance(val, BoolList):
        if not isinstance(input_val, list):
            return False
        return all(isinstance(x, bool) for x in input_val)
    elif isinstance(val, DoubleList):
        if not isinstance(input_val, list):
            return False
        return all(isinstance(x, float) for x in input_val)
    elif isinstance(val, (TensorList, OptionalTensorList)):
        if not isinstance(input_val, list):
            return False
        return all(isinstance(x, torch.Tensor) for x in input_val)
    elif isinstance(val, Null):
        raise TypeError("Setting a value where there should be a Null")


def resolve_op(operator: Operator) -> torch._ops.OpOverload:
    # pattern matching out the namespace and operation name
    namespace, op_name = operator.name.split("::")
    op = getattr(getattr(getattr(torch.ops, namespace), op_name), operator.overload)
    return op


def make_operators_list(
    execution_plan: ExecutionPlan,
) -> List[torch._ops.OpOverload]:
    operator_list = [resolve_op(operator) for operator in execution_plan.operators]
    return operator_list


class Interpreter:
    def __init__(self, program: Program) -> None:
        # Currently there is only 1 execution plan in the list -- this assert will help
        # catch any changes in the future
        assert len(program.execution_plan) == 1
        self.execution_plan: exir.schema.ExecutionPlan = program.execution_plan[0]
        self.container_metatype: exir.schema.ContainerMetadata = program.execution_plan[
            0
        ].container_meta_type

        # create buffer in memory and get reference to it
        # pyre-ignore
        self.data_buffers: List[bindings.DataBuffer] = [
            # pyre-ignore
            bindings.DataBuffer(b.storage, len(b.storage))
            for b in program.constant_buffer
        ]

        # generate the list of values (including tensors) and operators from the execution plan
        self._value_list: List[ValueType] = [
            Uninitialized() for val in self.execution_plan.values
        ]
        self._operators_list: List[torch._ops.OpOverload] = make_operators_list(
            self.execution_plan
        )

    def get_value_list(self) -> List[ValueType]:
        # TODO(meghajain) may need to change deepcopy to clone
        return copy.deepcopy(self._value_list)

    def get_operators_list(self) -> List[torch._ops.OpOverload]:
        return self._operators_list

    def get_constant_tensors(self) -> List[Tensor]:
        """
        No side effects on Interpreter's value list. List of constant tensors returned
        without having to run program.
        """
        tensors = []
        for elem in self.execution_plan.values:
            val = elem.val
            if isinstance(val, Tensor) and val.data_buffer_idx != 0:
                # load val into res
                # pyre-fixme[16]
                tensor = bindings.convert_to_tensor(
                    self.data_buffers[val.data_buffer_idx],
                    val.scalar_type,
                    val.sizes,
                    stride_from_dim_order(val.sizes, val.dim_order),
                )
                tensors.append(tensor)
        return tensors

    def load_value(self, idx: int) -> None:
        """
        Given an index in the value list, if value is `Uninitialized` or is a mutable object,
        like a Tensor List, calls `load` to load and initialize value into Interpreter's value_list.

        Args:
        `idx` : index in value lists that we want to load

        Returns: No returned values - value list is updated in place
        """
        # if instance of any mutable object, load regardless of being initialized
        if isinstance(
            self.execution_plan.values[idx].val, (TensorList, OptionalTensorList)
        ) or isinstance(self._value_list[idx], Uninitialized):
            self.load_from_value_list(idx)

    def load_from_value_list(self, idx: int) -> None:  # noqa
        """
        Accesses Execution Plan's value list at same index (see schema.py) to
        load and initialize value into Interpreter's value list. Extracts the
        necessary values depending on the type of the value. E.g. Tensor Lists
        have indices into the value list, so they are recursively loaded.
        If an Evalue is a Constant Tensor (denoted by allocation_info=None), it
        converts the python obj to a torch tensor object.

        Args:
        `idx` : index in value lists that we want to load

        Returns: No returned values - value list is updated in place
        """
        assert idx >= 0
        val = self.execution_plan.values[idx].val

        # Case through all possible Evalue Types
        if isinstance(val, Int):
            self._value_list[idx] = val.int_val
        elif isinstance(val, Bool):
            self._value_list[idx] = val.bool_val
        elif isinstance(val, Double):
            self._value_list[idx] = val.double_val
        elif isinstance(val, String):
            self._value_list[idx] = val.string_val
        elif isinstance(val, (IntList, BoolList, DoubleList)):
            unboxed_list = []
            for item in val.items:
                assert isinstance(item, int)
                assert isinstance(self.execution_plan.values[item].val, Int)
                # pyre-fixme [16] Undefined attribute [16]: Item `Bool` has no
                # attribute `int_val`.
                unboxed_list.append(self.execution_plan.values[item].val.int_val)
            self._value_list[idx] = unboxed_list
        elif isinstance(val, (TensorList, OptionalTensorList)):
            tensor_list = []
            for i in val.items:
                if i == -1:
                    tensor_list.append(None)
                    continue
                self.load_value(i)
                tensor_list.append(self._value_list[i])
            self._value_list[idx] = tensor_list
        elif isinstance(val, Tensor):
            if val.data_buffer_idx == 0:
                # TODO(zhengxu) Verify that argument is actually an out variant
                self._value_list[idx] = torch.empty(
                    val.sizes, dtype=get_scalar_type(val.scalar_type)
                )
            else:
                # Constant Tensor conversion
                # pyre-fixme [16]
                tensor = bindings.convert_to_tensor(
                    self.data_buffers[val.data_buffer_idx],
                    val.scalar_type,
                    val.sizes,
                    stride_from_dim_order(val.sizes, val.dim_order),
                )
                self._value_list[idx] = tensor
        elif isinstance(val, Null):
            self._value_list[idx] = None
        else:
            raise TypeError(
                f"Unexpected type, {type(val)}, with value, {val}, in Execution Plan values."
            )

        assert not isinstance(self._value_list[idx], Uninitialized)

    def set_value(self, idx: int, input_val: ValueType) -> None:
        """
        Given an index in the value list, and a value, updates
        Interpreter's value in value list at given index in place
        If value is meant to be a TensorList at given index,
        iterate through all the indices in the TensorList and
        update each placeholder with the given Tensor from `input_val`.

        Args:
        `idx` : index in value lists that we want to set

        `input_val` : value we want to put at `self._value_list[idx]`

        Returns: No returned values - value list is updated in place
        """
        evalue = self.execution_plan.values[idx]
        val = evalue.val

        if not comp_types(val, input_val):
            raise TypeError(
                f"Program trying to set a value of {input_val} : {type(input_val)} in memory location where {type(val)} is expected."
            )

        # Case through all possible Evalue Types
        if isinstance(
            val,
            (Int, Bool, Double, String, IntList, BoolList, DoubleList, Tensor, Null),
        ):
            self._value_list[idx] = input_val
        elif isinstance(val, (TensorList, OptionalTensorList)):
            assert isinstance(input_val, List)
            assert len(val.items) == len(input_val)
            tensor_list = []
            for i in range(len(val.items)):
                val_idx = val.items[i]
                self._value_list[val_idx] = input_val[i]
                tensor_list.append(input_val[i])
            self._value_list[idx] = tensor_list
        else:
            raise TypeError(
                f"Unexpected type, {type(val)}, with value, {val}, in Execution Plan values."
            )

    def call_kernel(self, kernel: KernelCall) -> None:
        """
        Calls operator from kernel:
        1. Determines kernel's operation through kernel.op_index,
           which indexes into operator list from Program's execution plan.
        2. After identifying operation, determines number of arguments and
        keyword arguments through operator schema.
        3. Extracts arguments from value list and calls operator.
        4. Sets the given output indices in value list with the values
           returned from operation.

        Args:
        `kernel` : stores information about operator and which indices in
                   the value list contain the necessary arguments

        Returns: No returned values - value list is updated with outputs
                from operator in place
        """

        operator = self._operators_list[kernel.op_index]
        num_args = len(
            [arg for arg in operator._schema.arguments if not arg.kwarg_only]
        )
        kwarg_list = [kwarg for kwarg in operator._schema.arguments if kwarg.kwarg_only]
        num_kwargs = len(kwarg_list)

        # Extracting arguments and keyword arguments from value_list given indices kernel.args
        args = []
        for i in kernel.args[:num_args]:
            self.load_value(i)
            args.append(self._value_list[i])

        kwargs = {}
        for j in range(num_kwargs):
            i = kernel.args[num_args + j]
            keyword = kwarg_list[j].name

            self.load_value(i)
            kwargs[keyword] = self._value_list[i]

        res = operator(*args, **kwargs)
        output_idxs = kernel.args[num_args + num_kwargs :]

        assert (
            len(output_idxs) == 1
        ), "emitter is expected to pack multiple outputs into a TensorList"
        if isinstance(res, tuple):
            self.set_value(output_idxs[0], list(res))
        else:
            self.set_value(output_idxs[0], res)

    def run(self, *raw_args: torch.Tensor) -> PyTree:
        """
        Loops through instructions given some inputs

        Args:
        `args` : list of inputs required for interpretation

        Returns:
        Outputs after completing all computations
        """

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        args, pytree = ex_pytree.tree_flatten((raw_args, {}))

        if pytree.to_str() != self.container_metatype.encoded_inp_str:
            raise TypeError(
                f"Arguments provided do not match required type. \nRequired: {self.container_metatype.encoded_inp_str} \nProvided: {pytree.to_str()}"
            )

        # Initialize user inputs in value list
        if len(self.execution_plan.inputs) != len(args):
            raise RuntimeError(
                f"Incorrect number of arguments provided. Expected {len(self.execution_plan.inputs)} values, but received {len(args)}"
            )
        for i in range(len(self.execution_plan.inputs)):
            idx = self.execution_plan.inputs[i]
            self._value_list[idx] = args[i]

        assert len(self.execution_plan.chains) == 1
        chain = self.execution_plan.chains[0]

        # instruction pointer
        ip = 0

        # Kernel loop
        while ip < len(chain.instructions):
            instruction = chain.instructions[ip]
            if isinstance(instruction.instr_args, KernelCall):
                self.call_kernel(instruction.instr_args)
            elif isinstance(instruction.instr_args, JumpFalseCall):
                self.load_value(instruction.instr_args.cond_value_index)
                ip = (
                    ip + 1
                    # pyre-ignore
                    if self._value_list[instruction.instr_args.cond_val_index]
                    # pyre-ignore
                    else instruction.instr_args.destination_instruction
                )
                continue
            elif isinstance(instruction.instr_args, MoveCall):
                move_to = instruction.instr_args.move_to
                move_from = instruction.instr_args.move_from
                self.load_value(move_from)
                self.load_value(move_to)
                self._value_list[move_to] = self._value_list[move_from]
            else:
                raise RuntimeError(
                    f"Received unknown instruction from program: {instruction}."
                )
            ip += 1

        ret = [self._value_list[i] for i in self.execution_plan.outputs]
        # pyre-fixme[16]: Module `pytree` has no attribute `from_str`.
        treespec = ex_pytree.from_str(self.container_metatype.encoded_out_str)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_unflatten`.
        return ex_pytree.tree_unflatten(ret, treespec)
