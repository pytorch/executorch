# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Takes an ExportedArtifact, or a collection of ExportedArtifacts, in execution dialect, and turns
them into a single ExecuTorch Program.

The provided ExportedArtifact's graph modules are in execution dialect and the emitter parses and
converts them into executorch instructions. The emitter walks the provided graphs and as it
encounters concrete values such as tensors or ints, it converts them to the serialized format and
stores them in a list for later use. The emitter walks the graph by traversing fx.nodes, these can
come in a variety of forms and are the primitives of execution at the graph module level. The most
common 3 we care about are 'call_function', 'place_holder', and 'output'. 'placeholder' and 'output'
handle io for the module and 'call_function' handles everything else. Within 'call_function' we may
encounter an operator or delegate call, in such case we parse the schema and emit all the inputs and
outputs (unless they have already previously been emitted), and then we convert the actual function
call into an executorch instruction such as KernelCall or DelegateCall.

When control flow is present in the graphmodule it will take the form of a few different types of
'call_function'. Today (June 14th 2023) only cond and map are supported. The actual operations of
these, such as the true/false branches of cond, or the mapping function of map, are stored as sub
graphmodules. When these are encountered during emission, the emitter will recursively emit them and
their instructions.
"""
# TODO(jakeszwe): add information here about how weights and other parameters are handled in the
# presence of aot autograd param lifting.

# pyre-strict
import ctypes
import hashlib
import operator
import typing
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, List, Mapping, Optional, Tuple, Union

import executorch.exir.memory as memory
import executorch.extension.pytree as ex_pytree
import torch
import torch.fx
from executorch.exir.delegate import executorch_call_delegate, is_lowered_module
from executorch.exir.dialects.backend._ops import BackendOpOverload
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.operator.convert import is_out_variant
from executorch.exir.passes.executorch_prim_ops_registry import is_sym_op
from executorch.exir.print_program import _stacktrace_to_framelist, inspect_node
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Bool,
    BoolList,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    DelegateCall,
    Double,
    DoubleList,
    EValue,
    ExecutionPlan,
    FreeCall,
    Instruction,
    Int,
    IntList,
    JumpFalseCall,
    KernelCall,
    MoveCall,
    Null,
    Operator,
    OptionalTensorList,
    ScalarType,
    String,
    Tensor,
    TensorList,
    TensorShapeDynamism,
)
from executorch.exir.tensor import (
    AddressSpaceOverflowException,
    layout_enum,
    make_allocation_info,
    make_tensor_value,
    memory_format_enum,
    scalar_type_enum,
    TensorSpec,
)
from executorch.exir.types import LeafValueSpec, ValueSpec
from torch._subclasses.fake_tensor import FakeTensor

from torch.export.exported_program import ExportedProgram
from torch.utils import _pytree as pytree

from typing_extensions import TypeAlias


@dataclass
class _ProgramState:
    """State shared between all methods of a program and the graph module it represents.

    Initialized once within emit_program and then shared across each entry point as they are
    emitted.
    """

    # Parallel list of specs and the buffers that backed them, have to add + 1 to any index in here
    # as index 0 in the constant_buffer is reserved.
    allocated_specs: List[TensorSpec] = field(default_factory=list)
    # Weights in any arbitrary graph_module only need to compare against weights from previously
    # emitted graph modules, not any weights emitted from itself. This should speed up the lookup,
    # from O(N) to O(1)
    cached_spec_hash_values: Dict[str, int] = field(default_factory=dict)
    cached_spec_mutable_hash_values: Dict[str, int] = field(default_factory=dict)
    # The 0 index is reserved to be pointed to by non-constant tensors, so add an empty placeholder.
    constant_buffer: List[Buffer] = field(default_factory=lambda: [Buffer(storage=b"")])
    # The 0 index is reserved to be pointed to by non-constant tensors, so add an empty placeholder.
    mutable_buffer: List[Buffer] = field(default_factory=lambda: [Buffer(storage=b"")])
    # Delegate data stored directly in the flatbuffer. Pointed to by BackendDelegateDataReference,
    # and should be copied to Program.backend_delegate_data.
    backend_delegate_data: List[BackendDelegateInlineData] = field(default_factory=list)


@dataclass
class _EmitterState:
    """State of a single emitter.

    Local to at least the entry point, and may be local to a subgraph of an entry point originating
    from control flow.
    """

    values: List[EValue]
    operators: List[Operator]
    delegates: List[BackendDelegate]
    operator_cache: Dict[Tuple[str, str], int]
    delegate_cache: Dict[bytes, int]
    emit_stacktrace: bool

    spec2id_dict: Dict[TensorSpec, int] = field(default_factory=dict)

    def spec2id(self, spec: TensorSpec) -> int:
        """Map a TensorSpec to value index in the values array."""
        assert spec in self.spec2id_dict, f"Spec is not found: {spec.debug()}"
        return self.spec2id_dict[spec]


@dataclass
class _AbstractValue:
    """Represents an already emitted EValue"""

    # Index in the values table of this EValue.
    id: int

    # Used for sanity checks for functions that expect to only receive AbstractValues.
    tensor: Optional[Tensor]


_EmitterValue: TypeAlias = Union[
    _AbstractValue, List[_AbstractValue], Tuple[_AbstractValue, ...]
]

_PythonValue: TypeAlias = Union[bool, int, float, torch.Tensor, List["_PythonValue"]]
_SchemaType: TypeAlias = Union[
    torch.OptionalType,
    torch.ListType,
    torch.FloatType,
    torch.BoolType,
    torch.IntType,
    torch.StringType,
    torch.TensorType,
]

_Target: TypeAlias = Union[Callable[..., _PythonValue], str]

_Argument: TypeAlias = Union[
    _EmitterValue,
    Tuple["_Argument", ...],
    List["_Argument"],
    Dict[str, "_Argument"],
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.memory_format,
    torch.layout,
    None,
]

_DelegateDebugIdentifierMap: TypeAlias = Union[
    Dict[int, Tuple[int]], Dict[str, Tuple[int]]
]


# pyre-ignore[13]: Attribute `node` is never initialized.
class _Emitter(torch.fx.Interpreter):
    """An abstract interpreter (https://wiki.mozilla.org/Abstract_Interpretation) used to emit the
    given traced torch.fx.GraphModule to the flatbuffer schema."""

    node: torch.fx.Node

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        emitter_state: _EmitterState,
        program_state: _ProgramState,
        instruction_start_offset: int = 0,
        binding_input_values: Optional[List[_AbstractValue]] = None,
        binding_output_values: Optional[List[_AbstractValue]] = None,
    ) -> None:
        super().__init__(graph_module)
        self.emitter_state = emitter_state
        self.program_state = program_state
        self.outputs: List[int] = []

        self.chain = Chain(
            inputs=[],
            outputs=[],
            instructions=[],
            stacktrace=None,
        )

        if "non_const_buffer_sizes" not in graph_module.meta.keys():
            raise RuntimeError(
                "Must set 'non_const_buffer_sizes' in graph meta in memory planning pass"
            )
        self.instruction_start_offset = instruction_start_offset
        self.binding_input_values = binding_input_values
        self.binding_output_values = binding_output_values
        self.graph_module: torch.fx.GraphModule = graph_module
        self.nodes: List[torch.fx.Node] = list(self.graph_module.graph.nodes)

        # Marks the placeholder node with its order so that we can match them with the corresponding
        # Abstract Value coming from top level.
        self.placeholder_count = 0

        self.concrete_output_ids: List[_AbstractValue] = []
        self.debug_handle_map: Dict[int, Union[int, List[int]]] = {}
        self.instr_id_to_delegate_debug_id_map: Dict[
            int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]
        ] = {}

    def _emit_node_specific_error(self, node: torch.fx.Node, err_msg: str) -> str:
        """Returns 'err_msg' with node specific information attached."""
        err_msg = f"Failed with error: {str(err_msg)}\n" + inspect_node(
            self.graph_module.graph, node
        )
        return err_msg

    def _internal_assert_emitter(
        self, pred: bool, node: torch.fx.Node, assert_msg: str
    ) -> None:
        """If pred is False, construct and raise a node specific error message."""
        if not pred:
            raise InternalError(self._emit_node_specific_error(node, assert_msg))

    def _emit_int_list(self, val: List[_Argument]) -> EValue:
        """Emits a list of integers as a collection of EValues.

        For every argument in 'val':
            - If it is a concrete value, then emit it and then place its location in the boxed list
            - If it is already an abstract value, then just place its location in the boxed list

        Int lists are boxed to handle symints whose values are determined at runtime, but could
        still end up inside lists for ops like view_copy(Tensor self, SymInt[] size)
        """
        boxed_list = []
        for item in val:
            if isinstance(item, _AbstractValue):
                boxed_list.append(item.id)
            elif isinstance(item, int):
                boxed_list.append(
                    self._emit_evalue(self._constant_to_evalue(item, None)).id
                )
            else:
                self._internal_assert_emitter(
                    False, self.node, "Unsupported type encountered in int list."
                )

        return EValue(IntList(boxed_list))

    def _emit_list(self, val: List[_Argument], val_type: _SchemaType) -> EValue:
        """Emits a list type.

        Emits the list stored in val. If the list is of Tensors, Optionals, or Ints the emitted list
        is boxed, otherwise the values are constant at runtime and stored inline.

        NOTE: When symbool and symfloat are supported bool and float lists will be stored boxed.
        """

        if isinstance(val_type, torch.BoolType):
            return EValue(BoolList(typing.cast(List[bool], val)))

        if isinstance(val_type, torch.IntType):
            return self._emit_int_list(val)

        if isinstance(val_type, torch.FloatType):
            return EValue(DoubleList(typing.cast(List[float], val)))

        if isinstance(val_type, torch.TensorType):
            values = []
            for v in val:
                assert isinstance(v, _AbstractValue)
                self._internal_assert_emitter(
                    v.tensor is not None,
                    self.node,
                    "AbstractValue corresponding to tensor type doesn't contain tensor value",
                )
                values.append(v.id)
            return EValue(TensorList(values))

        if isinstance(val_type, torch.OptionalType):
            # refine further
            actual_type = val_type.getElementType()
            if isinstance(actual_type, torch.TensorType):
                vals = []
                for v in val:
                    if v is None:
                        vals.append(-1)
                    else:
                        assert isinstance(v, _AbstractValue)
                        vals.append(v.id)
                return EValue(OptionalTensorList(vals))

        raise ExportError(
            ExportErrorType.NOT_SUPPORTED, f"Unknown list type: {val_type}"
        )

    def _tensor_spec_to_evalue(self, spec: TensorSpec) -> EValue:
        """Constructs an EValue from the given TensorSpec."""

        allocation_info = None
        buffer_idx = 0

        # Need to memory plan
        # Some users set mem_id on all tensors and then rely on the
        # default algos to set offsets, so need to check both.
        if spec.mem_id is not None and spec.mem_offset is not None:
            # Tensor is an activation.
            self._internal_assert_emitter(
                isinstance(spec.mem_id, int) and spec.mem_id >= 0,
                self.node,
                f"Non-const tensor should be an activation tensor: mem_id {spec.mem_id}",
            )

            self._internal_assert_emitter(
                isinstance(spec.mem_offset, int) and spec.mem_offset >= 0,
                self.node,
                f"Non-const tensor should be an activation tensor: mem_offset {spec.mem_offset}",
            )
            try:
                allocation_info = make_allocation_info(spec.mem_id, spec.mem_offset)
            except AddressSpaceOverflowException as e:
                raise InternalError(
                    self._emit_node_specific_error(
                        self.node,
                        (
                            f"{e}\nHint: If you are using a memory pass based on dynamic shape bounds, "
                            f"such as ConstraintBasedSymShapeEvalPass, this may be the cause of an "
                            f"unbacked SymInt with its upper bound lazily set to 2^64-1 (uint64 max) "
                            "during torch.export()."
                        ),
                    )
                )

        if spec.const:
            # Tensor with a blob we need to serialize. May not actually be constant at runtime
            # if it's a weight with an associated gradient
            spec_array_type = (
                ctypes.c_char * typing.cast(torch.UntypedStorage, spec.storage).nbytes()
            )

            buffer_data = (
                bytes(
                    ctypes.cast(
                        typing.cast(torch.UntypedStorage, spec.storage).data_ptr(),
                        ctypes.POINTER(spec_array_type),
                    ).contents
                )
                if spec.allocated_memory != 0
                else b""
            )

            hashed = hashlib.sha256(buffer_data).hexdigest()

            if allocation_info:
                buffer_idx = self.program_state.cached_spec_mutable_hash_values.get(
                    hashed, -1
                )
            else:
                buffer_idx = self.program_state.cached_spec_hash_values.get(hashed, -1)

            # Haven't seen this constant before
            if buffer_idx == -1:
                # Update buffer_idx to point to the end of the list where we are adding the new buffer.
                buffer = Buffer(storage=buffer_data)
                self.program_state.allocated_specs.append(spec)
                # +1 because the first buffer location is reserved

                if allocation_info:
                    buffer_idx = len(self.program_state.mutable_buffer)
                    self.program_state.cached_spec_mutable_hash_values[hashed] = (
                        buffer_idx
                    )
                    self.program_state.mutable_buffer.append(buffer)
                else:
                    buffer_idx = len(self.program_state.constant_buffer)
                    self.program_state.cached_spec_hash_values[hashed] = buffer_idx
                    self.program_state.constant_buffer.append(buffer)

            if spec.const and spec.nbytes() != len(buffer_data):
                raise InternalError(
                    self._emit_node_specific_error(
                        self.node,
                        f"Tensor spec has buffer of size {len(buffer_data)}, but expected nbytes of {spec.nbytes()}",
                    )
                )

        # For constant tensors, allocation_info = None.
        return EValue(make_tensor_value(buffer_idx, allocation_info, spec))

    def _get_list_tuple_jit_type(
        self, val: Union[Tuple[_Argument], List[_Argument]]
    ) -> _SchemaType:
        """Returns the JIT type for the given python type."""
        assert isinstance(
            val, (list, tuple)
        ), f"Input to _get_list_tuple_jit_type was expected to be an instance of a list or tuple but received {type(val)}"
        is_tensor_type = all(
            isinstance(v, _AbstractValue) and v.tensor is not None for v in val
        )
        if is_tensor_type:
            return torch.TensorType.get()
        elif isinstance(val[0], int):
            return torch.IntType.get()
        elif isinstance(val[0], bool):
            return torch.BoolType.get()
        elif isinstance(val[0], float):
            return torch.FloatType.get()

        raise InternalError(
            self._emit_node_specific_error(
                self.node,
                "Couldn't determine JitType for list/tuple of elements. Only supports int, float, bool, and Tensor.",
            )
        )

    def _constant_to_evalue(  # noqa: C901
        self,
        val: _Argument,
        val_type: Optional[_SchemaType],
    ) -> EValue:
        """Converts a constant value to an EValue.

        Returns an EValue given the Python representation and JIT type. On common paths there should
        always be a JIT type provided. Users can pass in a None to infer the JIT type but this
        should never be the default case due to the existence of container types.
        """
        if val is None:
            return EValue(Null())

        if isinstance(val, (list, tuple)):
            # Refine Optional[List[T]] -> List[T] This works because if the val was None it would
            # have converted to Null before this function call.
            if val_type is None:
                val_type = torch.ListType(
                    self._get_list_tuple_jit_type(val)  # pyre-ignore
                )
            if isinstance(val_type, torch.OptionalType):
                val_type = val_type.getElementType()
            assert isinstance(val_type, torch.ListType)
            return self._emit_list(
                typing.cast(List[_Argument], val),
                typing.cast(_SchemaType, val_type.getElementType()),
            )

        if isinstance(val, float):
            return EValue(Double(val))

        if isinstance(val, bool):
            return EValue(Bool(val))

        if isinstance(val, int):
            return EValue(Int(val))

        if isinstance(val, str):
            return EValue(String(val))

        if isinstance(val, torch.dtype):
            return EValue(Int(scalar_type_enum(val)))

        if isinstance(val, torch.layout):
            return EValue(Int(layout_enum(val)))

        if isinstance(val, torch.memory_format):
            try:
                return EValue(Int(memory_format_enum(val)))
            except KeyError:
                raise InternalError(
                    self._emit_node_specific_error(
                        self.node,
                        f"Tensor has a memory_format that is unsupported in ExecuTorch: {val}",
                    )
                )

        if isinstance(val, torch.Tensor):
            raise ExportError(
                ExportErrorType.NOT_SUPPORTED,
                self._emit_node_specific_error(
                    self.node,
                    "constant_to_evalue should not be encountering constant tensors, they should be emitted through other codepaths.",
                ),
            )

        raise ExportError(
            ExportErrorType.NOT_SUPPORTED,
            self._emit_node_specific_error(
                self.node, f"Unsupported constant type: {type(val).__name__}"
            ),
        )

    def _emit_evalue(self, val: EValue) -> _AbstractValue:
        """Writes an EValue to the emitter state.

        Given an Evalue, adds it to the emitter_state's values table, and returns the AbstractValue
        representing it.
        """
        self.emitter_state.values.append(val)
        tensor = val.val if isinstance(val.val, Tensor) else None
        return _AbstractValue(len(self.emitter_state.values) - 1, tensor)

    def _emit_spec(self, spec: ValueSpec) -> _EmitterValue:
        """Given the provided spec constructs the corresponding EValue from it and then emits it."""

        def _process(spec: LeafValueSpec) -> _AbstractValue:
            if isinstance(spec, (list, tuple)):
                raise InternalError(
                    self.emit_node_specific_error(
                        self.node,
                        "Node spec should be either non-nested container or a scalar object",
                    )
                )

            # ScalarSpec can theoretically be handled fine, but it shouldn't be appearing so rather
            # than handle it, assert that it isn't supposed to be present. In the future if it has a
            # reason to appear we can relax this assert.
            self._internal_assert_emitter(
                isinstance(spec, TensorSpec),
                self.node,
                f"Invalid node spec expected TensorSpec received {spec}",
            )

            ret = self._emit_evalue(self._tensor_spec_to_evalue(spec))  # pyre-ignore
            self.emitter_state.spec2id_dict[spec] = ret.id  # pyre-ignore
            return ret

        return pytree.tree_map(_process, spec)

    def _merge_chain(self, chain: Chain) -> None:
        """Merges the chain generated from subgraphs (like those originating from control flow) back
        into the main program chain."""
        self.chain.instructions.extend(chain.instructions)

    def _emit_cond(
        self,
        args: Tuple[_Argument, ...],
        subemitter_binding_output_values: Optional[List[_AbstractValue]],
    ) -> List[_AbstractValue]:
        """Emits control_flow.cond.

        Converts the higher order op into jumps and inlines the submodules of the true and false
        branches. Control flow can be nested. The general emitted structure is: <Jump Instruction> -
        decides which branch to take <True Branch> <Jump Instruction> - jumps to End Of Cond after
        executing true branch <False Branch> <End Of Cond>
        """
        pred, true_branch, false_branch, inputs = args

        # Emit the true branch.
        assert isinstance(true_branch, torch.fx.GraphModule)
        true_branch_emitter = _Emitter(
            true_branch,
            self.emitter_state,
            self.program_state,
            instruction_start_offset=self.instruction_start_offset
            + len(self.chain.instructions)
            + 1,
            binding_input_values=typing.cast(List[_AbstractValue], inputs),
            binding_output_values=subemitter_binding_output_values,
        )
        true_branch_emitter.run()

        # Emit the jump.
        assert isinstance(pred, _AbstractValue)
        jf_instruction_to_skip_true = Instruction(
            JumpFalseCall(
                cond_value_index=pred.id,
                destination_instruction=self.instruction_start_offset
                + len(self.chain.instructions)
                + len(true_branch_emitter.chain.instructions)
                # This jump instruction should point at instruction that is after all instructions
                # for the true branch. The reason we add 2 is because we need to account for this
                # instruction we are creating right now and the jump instruction that true branch
                # will create.
                + 2,
            )
        )

        # Insert the branch picking jump instruction to the main chain.
        self.chain.instructions.append(jf_instruction_to_skip_true)
        # Now that we created the true branch instructions, we move them to the main chain.
        self._merge_chain(true_branch_emitter.chain)

        # emit false branch
        assert isinstance(false_branch, torch.fx.GraphModule)
        false_branch_emitter = _Emitter(
            false_branch,
            self.emitter_state,
            self.program_state,
            instruction_start_offset=self.instruction_start_offset
            + len(self.chain.instructions)
            + 1,
            binding_input_values=typing.cast(List[_AbstractValue], inputs),
            binding_output_values=subemitter_binding_output_values,
        )
        false_branch_emitter.run()

        # We bake in constant False because this will trigger the instruction to jump over all false
        # branch instructions and point at the start of instruction right after control flow.
        value = self._emit_evalue(EValue(Bool(False)))
        jf_instruction_to_skip_false = Instruction(
            JumpFalseCall(
                cond_value_index=value.id,
                destination_instruction=self.instruction_start_offset
                + len(self.chain.instructions)
                + len(false_branch_emitter.chain.instructions)
                + 1,
            )
        )
        self.chain.instructions.append(jf_instruction_to_skip_false)
        self._merge_chain(false_branch_emitter.chain)
        return subemitter_binding_output_values

    def _emit_map(
        self,
        args: Tuple[_Argument, ...],
        subemitter_binding_output_values: List[_AbstractValue],
    ) -> List[_AbstractValue]:
        """Emits torch.map.

        Converts the higher order op into a loop constructed from jump instructions and primitive
        int operations. A concat-like custom op is also injected into the submodule code to handle
        the construction of the map output.

        Below is what the input graph that is provided to emit_map looks like. class
        TestMapCond(torch.nn.Module): def __init__(self):
            super().__init__()

        def forward(self, x,y):
            return control_flow.map(map_fn, x, y)

        Corresponding graph: def forward(self, arg0_1, arg1_1):
            submodule_0 = self.submodule_0 map_1 = torch.ops.higher_order.map_impl(submodule_0, arg0_1, arg1_1);
            submodule_0 = arg0_1 = arg1_1 = None return [map_1]

        submodule_0: def forward(self, arg0_1, arg1_1):
            add_tensor = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None return
            add_tensor

        Post the transformations done by emit_map this is what the submodule program looks like. def
        forward(self, arg0_1, arg1_1):
            sym_size = torch.ops.aten.sym_size(arg0_1) # Emitter creates a variable here to track
            iteration index select_copy_tensor = torch.ops.aten.select(arg0_1, 0, iteration_index)
            add_tensor = torch.ops.aten.add.Tensor(select_copy_tensor, arg1_1);  arg0_1 = arg1_1 =
            None output_of_map = torch.ops.executorch.prim.et_copy_index(output_of_map, add_tensor,
            iteration_index) iteration_index = torch.ops.executorch.prim.add.int(iteration_index, 1,
            iteration_index) done_bool = torch.ops.executorch.prim.eq.int(iteration_index, sym_size,
            done_bool) # Emitter inserts a instruction here, if done_bool == False jump to
            selcect_copy op # if not continue. return add_tensor
        """
        assert isinstance(
            subemitter_binding_output_values, (list, tuple)
        ), f"Expect a list for subemitter_binding_output_values for map. Got {subemitter_binding_output_values}."

        if len(subemitter_binding_output_values) != 1:
            raise RuntimeError(
                f"Multiple outputs are not supported. Got {len(subemitter_binding_output_values)}."
            )
        f, mapped_args, inputs = args
        assert isinstance(mapped_args, (list, tuple))
        num_mapped_args: int = len(mapped_args)
        if num_mapped_args != 1:
            raise RuntimeError(
                f"Emitting map with more than one mapped args is not supported. Got {num_mapped_args}."
            )
        x = mapped_args[0]

        assert isinstance(f, torch.fx.GraphModule)

        # Generate the EValue that we will use as our iterator index to keep track of which
        # iteration we are currently on.
        iter_idx = self._emit_evalue(EValue(Int(0)))
        # Generate the kernel call that will output the number of iterations we need to run for this
        # input tensor.
        op_index, op = self._get_operator(
            name="aten::sym_size",
            overload="int",
        )
        sym_size = self._emit_evalue(EValue(Int(0)))
        kernel = Instruction(
            KernelCall(
                op_index=op_index,
                args=[x.id, self._emit_evalue(EValue(Int(0))).id, sym_size.id],
            )
        )
        self.chain.instructions.append(kernel)

        # This kernel call will slice the input tensor along the index specified in iter_idx to
        # generate the input slice on which this iteration will be working on.
        op_index, op = self._get_operator(
            name="aten::select_copy",
            overload="int_out",
        )
        # This select copy has to output to the tensor which is the input placeholder to the map
        # sub-graph. That placeholder isn't allocated an EValue id until the map emitter is run, so
        # we temporarily store -1 until the map emitter is run during which the placeholder will be
        # allocated an EValue id. After the map emitter is run we will retrieve that id and replace
        # the -1's.
        kernel = Instruction(
            KernelCall(
                op_index=op_index,
                args=[
                    x.id,
                    self._emit_evalue(EValue(Int(0))).id,
                    iter_idx.id,
                    -1,  # input_tensor_value.id,
                    -1,  # input_tensor_value.id,
                ],
            )
        )
        # Store the index of this instruction as it will be where we will jump back to after the end
        # of an iteration.
        jump_to_instruction = self.instruction_start_offset + len(
            self.chain.instructions
        )
        self.chain.instructions.append(kernel)

        # Emit the map operator submodule.
        map_emitter = _Emitter(
            f,
            self.emitter_state,
            self.program_state,
            instruction_start_offset=self.instruction_start_offset
            + len(self.chain.instructions),
            # Only the first input is a placeholder, rest of the inputs are args to the map fn.
            binding_input_values=[-1, *inputs],
            binding_output_values=subemitter_binding_output_values,
        )
        map_emitter.run()

        # Merge all the instructions from the map submodule.
        self._merge_chain(map_emitter.chain)
        # Get rid of the return instruction emitted by the map subemitter.
        self.chain.instructions.pop()
        # At the end of each submodule emit we insert a move call that moves the output of the
        # submodule to a deterministic EValue, which is especially useful for if/else branches where
        # we want the output of either branch to be in the same EValue, but we don't need a move
        # here as our custom op executorch_prim::et_copy_index which is inserted later does that
        # for us.

        # Now that the map emitter has finished running retrieve the input placeholder EValue id and
        # update the select_copy kernel call to output to those id's.
        kernel.instr_args.args[-1] = map_emitter.binding_input_values[0].id
        kernel.instr_args.args[-2] = kernel.instr_args.args[-1]

        self._internal_assert_emitter(
            len(map_emitter.concrete_output_ids) == 1,
            self.node,
            "Map should return only one element",
        )

        # Here we call the custom op, specially added for the map operator. The output of this
        # iteration will be appended to the accumulator tensor that we are maintaining. This
        # accumulator tensor is the actual output of the map submodule.
        op_index, op = self._get_operator(
            name="executorch_prim::et_copy_index",
            overload="tensor",
        )
        kernel = Instruction(
            KernelCall(
                op_index,
                args=[
                    subemitter_binding_output_values[0].id,
                    map_emitter.concrete_output_ids[0].id,
                    iter_idx.id,
                ],
            )
        )
        self.chain.instructions.append(kernel)

        # Increment iter_idx to mark that we have completed an iteration.
        op_index, op = self._get_operator(
            name="executorch_prim::add",
            overload="Scalar",
        )
        kernel = Instruction(
            KernelCall(
                op_index,
                args=[iter_idx.id, self._emit_evalue(EValue(Int(1))).id, iter_idx.id],
            )
        )
        self.chain.instructions.append(kernel)

        jump_bool_value = self._emit_evalue(EValue(Bool(False)))

        # Generate the kernel call to check whether or not we have completed all the iterations. If
        # not jump back to the select_copy instruction that we generated at the beginning of this
        # section.
        op_index, op = self._get_operator(
            name="executorch_prim::eq",
            overload="Scalar",
        )
        kernel = Instruction(
            KernelCall(
                op_index,
                args=[iter_idx.id, sym_size.id, jump_bool_value.id],
            )
        )
        self.chain.instructions.append(kernel)

        jf_beginning_loop = Instruction(
            JumpFalseCall(
                cond_value_index=jump_bool_value.id,
                destination_instruction=jump_to_instruction,
            )
        )

        self.chain.instructions.append(jf_beginning_loop)

        # Reset iter_idx in case we plan to run the model again.
        op_index, op = self._get_operator(
            name="executorch_prim::sub",
            overload="Scalar",
        )
        kernel = Instruction(
            KernelCall(
                op_index,
                args=[iter_idx.id, sym_size.id, iter_idx.id],
            )
        )
        self.chain.instructions.append(kernel)

        return subemitter_binding_output_values

    def _emit_control_flow(
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _EmitterValue:
        """Wraps common logic for emitting all control flow operations.

        See the more specific emission functions for more details on how cond or map get emitted.
        """
        subemitter_binding_output_values = pytree.tree_map(
            lambda spec: self._emit_spec(spec),
            self.node.meta["spec"],
        )

        if target is torch.ops.higher_order.cond:
            return self._emit_cond(args, subemitter_binding_output_values)
        elif target is torch.ops.higher_order.map_impl:
            return self._emit_map(args, subemitter_binding_output_values)
        else:
            raise InternalError(
                self._emit_node_specific_error(
                    self.node, f"Unsupported control flow operator: {target}"
                )
            )

    def _emit_view(self, args: Tuple[_Argument, ...]) -> _EmitterValue:
        assert len(args) == 2

        self_arg = self._emit_argument(args[0], torch.TensorType)  # pyre-ignore[6]
        size_arg = self._emit_argument(args[1], torch.ListType.ofInts())
        out_arg = self._emit_argument(
            self._emit_spec(self.node.meta["spec"]), torch.TensorType  # pyre-ignore[6]
        )

        op_idx, op = self._get_operator(
            name="executorch_prim::et_view",
            overload="default",
        )
        kernel = Instruction(
            KernelCall(
                op_idx,
                args=[
                    self_arg.id,
                    size_arg.id,
                    out_arg.id,
                ],
            )
        )
        self.chain.instructions.append(kernel)
        return out_arg

    def _add_debug_handle(
        self,
        emitter_id: int,
        target: _Target,
        # pyre-ignore[11]: Annotation `LoweredBackendModule` is not defined as a type.
        lowered_module: "Optional[LoweredBackendModule]" = None,  # noqa: F821
    ) -> None:
        """Updates the debug handle information for the current node.

        If the current node is a delegate we agregate the debug handles of the subgraph and store
        them in the map. If the current node is any other type we store the original information in
        the debug handle map and replace it with the executorch instruction index corresponding to
        this node.
        """
        # If it's a delegate call, collect the list of debug handles that are consumed by this
        # delegate call and store it in the debug handle map.
        if target == executorch_call_delegate:
            debug_handle_list = []
            # Use the lowered_module to fetch the original graph and its debug
            # handles.
            for node in lowered_module.original_module.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.meta.get("debug_handle") is not None
                ):
                    debug_handle_list.append(node.meta.get("debug_handle"))
            self.debug_handle_map[emitter_id] = debug_handle_list
            # Debug handle for this node is the emitter_id which is essentially the index of the
            # instruction in the chain.
            self.node.meta["debug_handle"] = emitter_id
            return

        if self.node.meta.get("debug_handle") is not None:
            # Store the original debug handle in the debug handle map.
            self.debug_handle_map[emitter_id] = self.node.meta.get("debug_handle")
            # Replace the debug handle in the metadata of the node with the emitter id which
            # represents the instruction index in the chain. We do this because in the runtime the
            # instruction index is what is logged during perf/debug data logging and hence we want
            # to store this in the node so that we can map the data logged by the runtime back to
            # the node.
            self.node.meta["debug_handle"] = emitter_id

    def _add_delegate_map(
        self,
        lowered_module: "LoweredBackendModule",  # noqa
        delegate_instruction_id: int,
    ) -> None:
        """
        Store the delegate map from this lowered module into the dictionary of delegate maps. It
        will later be used for various debugging purposes such as linking back to original source
        code, module hierarchy etc.
        """
        delegate_map = {}
        if hasattr(lowered_module, "meta"):
            delegate_map = lowered_module.meta.get("debug_handle_map", {})

        self.instr_id_to_delegate_debug_id_map[delegate_instruction_id] = {
            "name": lowered_module.backend_id,
            "delegate_map": delegate_map,
        }

    def _emit_argument(
        self, arg: _Argument, arg_type: Optional[_SchemaType]
    ) -> _AbstractValue:
        """Emit an argument to an operator or delegate if it had not already been emitted otherwise
        return the previously emitted location"""
        if isinstance(arg, _AbstractValue):
            return arg
        return self._emit_evalue(self._constant_to_evalue(arg, arg_type))

    def _get_sym_ret(
        self,
        val: Tuple[Union[torch.SymInt, torch.BoolType, torch.FloatType, FakeTensor]],
    ) -> Optional[_AbstractValue]:
        """
        Returns the emit ret for sym value.
        """
        ret = None
        if isinstance(val, torch.SymInt):
            ret = self._emit_evalue(EValue(Int(0)))
        elif isinstance(val, torch.BoolType):
            ret = self._emit_evalue(EValue(Bool(False)))
        elif isinstance(val, torch.FloatType):
            ret = self._emit_evalue(EValue(Double(0)))
        return ret

    def _get_sym_and_fake_tensor_ret(
        self,
        val: Tuple[Union[torch.SymInt, torch.BoolType, torch.FloatType, FakeTensor]],
        spec: TensorSpec,
    ) -> Union[List[_AbstractValue], _AbstractValue, Tuple[_AbstractValue, ...]]:
        # Try to get the ret if it's a sym value.
        ret = self._get_sym_ret(val)
        # If the ret is None, it means that the val is not a sym value, but a regular tensor
        if ret is None:
            ret = self._emit_spec(spec)
        assert ret is not None, "Can't have a None ret"
        return ret

    def _emit_delegate(
        self,
        lowered_module: "LoweredBackendModule",  # noqa
        args: Tuple[_Argument, ...],
        kwargs: Dict[str, _Argument],
    ) -> _EmitterValue:
        """Emit the delegates inputs and outputs as specified by the schema, then emit the
        delegate's blob."""
        processed_bytes = lowered_module.processed_bytes

        delegate_index = self.emitter_state.delegate_cache.get(processed_bytes)
        delegate_ret = None

        if isinstance(self.node.meta["spec"], list):
            delegate_ret = []
            for index, _ in enumerate(self.node.meta["val"]):
                ret = self._get_sym_and_fake_tensor_ret(
                    self.node.meta["val"][index], self.node.meta["spec"][index]
                )
                delegate_ret.append(ret)
        elif isinstance(self.node.meta["spec"], tuple):
            if isinstance(self.node.meta["val"], FakeTensor):
                # There is a case when node.meta["spec"] is (TensorSpec, ) while node.meta["val"] is FakeTensor
                ret = self._get_sym_and_fake_tensor_ret(
                    self.node.meta["val"], self.node.meta["spec"][0]
                )
                delegate_ret = (ret,)
            else:
                delegate_ret = []
                for index, _ in enumerate(self.node.meta["val"]):
                    ret = self._get_sym_and_fake_tensor_ret(
                        self.node.meta["val"][index], self.node.meta["spec"][index]
                    )
                    delegate_ret.append(ret)
                delegate_ret = tuple(delegate_ret)
        elif isinstance(self.node.meta["spec"], TensorSpec):
            ret = self._get_sym_and_fake_tensor_ret(
                self.node.meta["val"], self.node.meta["spec"]
            )
            delegate_ret = ret
        else:
            raise NotImplementedError(
                f"self.node.meta['spec'] {type(self.node.meta['spec'])} is not supported"
            )
        assert delegate_ret is not None, "Can't have a None delegate_ret"
        if delegate_index is None:
            # Allocate an entry for the data. TODO(T150113674): Reuse any duplicate entries if
            # present.
            data_index: int = len(self.program_state.backend_delegate_data)
            self.program_state.backend_delegate_data.append(
                BackendDelegateInlineData(data=processed_bytes)
            )

            backend_delegate = BackendDelegate(
                id=lowered_module.backend_id,
                processed=BackendDelegateDataReference(
                    location=DataLocation.INLINE, index=data_index
                ),
                compile_specs=lowered_module.compile_specs,
            )
            delegate_index = len(self.emitter_state.delegate_cache)
            self.emitter_state.delegates.append(backend_delegate)
            self.emitter_state.delegate_cache[processed_bytes] = delegate_index

        # TODO(angelayi) Will need to emit the kwargs too, in the correct order according to the
        # function's spec and with default arguments. This requires us to store the function's spec
        # in to_backend()
        delegate_args = [
            self._emit_argument(arg, None).id
            for arg in typing.cast(List[_Argument], args)
        ]

        for elem in pytree.tree_flatten(delegate_ret)[0]:
            delegate_args.append(elem.id)

        self.chain.instructions.append(
            Instruction(DelegateCall(delegate_index=delegate_index, args=delegate_args))
        )

        return delegate_ret

    def _get_operator(self, name: str, overload: str) -> Tuple[int, Operator]:
        """Given a fully qualified name, lookups the operator in the ExecuTorch Program, or adds it
        if it is not already present"""
        key = (name, overload)
        op_index = self.emitter_state.operator_cache.get(key)
        if op_index is not None:
            return op_index, self.emitter_state.operators[op_index]

        op_index, operator = len(self.emitter_state.operators), Operator(
            name=name, overload=overload
        )
        self.emitter_state.operators.append(operator)
        self.emitter_state.operator_cache[key] = op_index
        return op_index, operator

    def _emit_operator(  # noqa: C901
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _EmitterValue:
        """Emits an operator (aten or custom), directly translates to a call_kernel instruction."""
        assert isinstance(
            target, (torch._ops.OpOverload, EdgeOpOverload, BackendOpOverload)
        ), f"target is {target}"

        # grab the name
        op_name = target._overloadpacket._qualified_op_name
        op_overload = ""
        if target._overloadname != "default":
            op_overload = target._overloadname

        def _get_empty_tensor_evalue() -> EValue:
            """Constructs an EValue for an empty tensor."""
            return EValue(
                Tensor(
                    scalar_type=ScalarType.BYTE,
                    # The runtime currently only supports tensors with offset 0.
                    storage_offset=0,
                    sizes=[0],
                    dim_order=[],
                    requires_grad=False,
                    layout=0,
                    data_buffer_idx=0,
                    allocation_info=None,
                    shape_dynamism=TensorShapeDynamism.STATIC,
                )
            )

        op_index, operator = self._get_operator(name=op_name, overload=op_overload)

        # Emit the args and kwargs in the order according to the function schema.
        kernel_args = []
        out_args = []
        for i, schema_arg in enumerate(target._schema.arguments):
            if schema_arg.name in kwargs:
                kernel_arg = kwargs[schema_arg.name]
            elif not schema_arg.kwarg_only and i < len(args):
                kernel_arg = args[i]
            else:
                # Emit default values
                kernel_arg = schema_arg.default_value

            if kernel_arg is None and isinstance(schema_arg.type, torch.TensorType):
                kernel_arg = self._emit_evalue(_get_empty_tensor_evalue())

            kernel_args.append(self._emit_argument(kernel_arg, schema_arg.type).id)

            if schema_arg.is_out:
                out_args.append((schema_arg.name, kernel_arg))

        if is_out_variant(op_name, op_overload):
            ret = [val for _, val in out_args]
            ret = ret[0] if len(ret) == 1 else ret
        elif is_sym_op(target):
            assert (
                len(target._schema.returns) == 1
            ), "Only returning a single Sym from symbolic ops is supported currently."
            assert type(target._schema.returns[0].type) in (
                torch.IntType,
                torch.FloatType,
                torch.BoolType,
                torch.NumberType,
            ), f"Only symbolic ops that return a Int Bool Float are supported currently got {type(target._schema.returns[0].type)}."
            ret = self._get_sym_ret(target._schema.returns[0])
            if ret is None:  # type(target._schema.returns[0].type) == torch.NumberType:
                # Cant definitively say what type this is, the runtime operator just overrides the EValue completely
                # though so we can just serialize whatever as a placeholder.
                ret = self._emit_evalue(EValue(Int(0)))
        else:
            ret = self._emit_spec(self.node.meta["spec"])

        out_args = (
            self._emit_evalue(
                EValue(TensorList([cast(_AbstractValue, val).id for val in ret]))
            )
            if isinstance(ret, list)
            else ret
        )

        for elem in pytree.tree_flatten(out_args)[0]:
            kernel_args.append(cast(_AbstractValue, elem).id)

        self.chain.instructions.append(
            Instruction(KernelCall(op_index=op_index, args=kernel_args))
        )
        self._add_debug_handle(len(self.chain.instructions) - 1, target)

        # Get the stacktrace if it exists for each instruction.
        if self.emitter_state.emit_stacktrace:
            stack_trace = self.node.meta["stack_trace"]
            chain_stacktrace = self.chain.stacktrace or []

            chain_stacktrace.append(_stacktrace_to_framelist(stack_trace))
            self._internal_assert_emitter(
                len(chain_stacktrace) == len(self.chain.instructions),
                self.node,
                f"Each instruction should have corresponding stacktrace received {len(self.chain.instructions)} \
                instructions and {len(chain_stacktrace)} stacktraces",
            )
            self.chain.stacktrace = chain_stacktrace

        return cast(_EmitterValue, ret)

    def _emit_free(self, spec: TensorSpec) -> _AbstractValue:
        """Emits a FreeCall instruction to release a given Unbounded Tensor's memory."""
        self.chain.instructions.append(
            Instruction(FreeCall(value_index=self.emitter_state.spec2id(spec)))
        )
        # The value is not used but the caller expects an AbstractValue returned.
        return _AbstractValue(None, None)  # pyre-ignore

    def _emit_prim_getters(self, prim_getters: Dict[str, Any]) -> List[ExecutionPlan]:
        """
        Given a mapping of function names to return values, emit simple execution
        plans that just return these constant values.

        Precondition: All the values are primitives (bool, float, int, str, enum)
        or structures (list, dict) of them.
        """
        plans = []
        # flatten any structures
        for method, vals in prim_getters.items():
            # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
            flattened_output, spec = ex_pytree.tree_flatten(vals)
            spec = spec.to_str()
            chain = Chain(
                inputs=[],
                outputs=[],
                instructions=[],
                stacktrace=None,
            )

            # switch on type of prim
            values = []
            for val in flattened_output:
                if isinstance(val, float):
                    values.append(EValue(Double(val)))

                elif isinstance(val, bool):
                    values.append(EValue(Bool(val)))

                elif isinstance(val, int):
                    values.append(EValue(Int(val)))

                elif isinstance(val, str):
                    values.append(EValue(String(val)))

                elif isinstance(val, torch.dtype):
                    values.append(EValue(Int(scalar_type_enum(val))))

                elif isinstance(val, torch.layout):
                    values.append(EValue(Int(layout_enum(val))))

                elif isinstance(val, torch.Tensor):
                    values.append(
                        self._tensor_spec_to_evalue(
                            TensorSpec.from_tensor(val, const=True)
                        )
                    )

                else:
                    raise ExportError(
                        ExportErrorType.NOT_SUPPORTED,
                        f"Error emitting {method} which returns a value of type {type(val)}. which is not a supported primitive",
                    )

            # add to plans
            plans.append(
                ExecutionPlan(
                    name=method,
                    values=values,
                    inputs=[],
                    outputs=list(range(0, len(values))),
                    chains=[chain],
                    operators=[],
                    delegates=[],
                    non_const_buffer_sizes=[0],
                    container_meta_type=ContainerMetadata("", spec),
                )
            )
        return plans

    def fetch_attr(self, target: _Target) -> _AbstractValue:
        """Fetch weights and other module parameters. If the attribute is a tensor, emit it."""
        attr = super().fetch_attr(target)  # pyre-fixme[6]

        if isinstance(attr, torch.Tensor):
            return self._emit_evalue(
                self._tensor_spec_to_evalue(TensorSpec.from_tensor(attr, const=True))
            )

        elif isinstance(attr, torch._C.ScriptObject):
            raise ExportError(
                ExportErrorType.NOT_SUPPORTED,
                f"Custom class {attr} is not supported in EXIR",
            )

        else:
            return attr

    def call_module(  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> None:
        """Unsupported in execution IR, so unhandled by the emitter."""
        raise InternalError(
            self._emit_node_specific_error(self.node, "call_module is not supported")
        )

    def call_method(  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _EmitterValue:
        """Unsupported in execution IR, so unhandled by the emitter."""
        raise InternalError(
            self._emit_node_specific_error(self.node, "call_method is not supported")
        )

    def placeholder(  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _AbstractValue:
        """Performs actions for the placeholder node of a graph module.

        The placeholder node of the top level entry point is handled by TopLevelEmitter. This
        function only executes on control flow subgraphs. Takes the inputs of the subgraph that had
        not previously been emitted and emits them.
        """
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        value = self.binding_input_values[self.placeholder_count]
        # This indicates that the placeholder wasn't allocated an EValue id before this sub-emitter
        # was run, so we generate one now.
        if value == -1:
            value = self._emit_evalue(
                self._tensor_spec_to_evalue(self.node.meta["spec"])
            )
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            self.binding_input_values[self.placeholder_count] = value
        self.placeholder_count += 1
        return value

    def output(  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> None:
        """Performs actions for the output node of a graph module.

        The output node of the top level entry point is handled by TopLevelEmitter. This function
        only executes on control flow subgraphs. Takes the outputs of the subgraph (if any) and
        inserts instructions to move them to the common output location between control flow
        branches.
        """
        self.concrete_output_ids = list(pytree.tree_flatten(args[0])[0])
        binding_output_values = self.binding_output_values
        if binding_output_values is not None:
            binding_output_list, _ = pytree.tree_flatten(binding_output_values)

            self._internal_assert_emitter(
                len(binding_output_list) == len(self.concrete_output_ids),
                self.node,
                "The number of binding output values should match the args to output",
            )

            for move_from, move_to in zip(
                self.concrete_output_ids, binding_output_list
            ):
                if move_from != move_to:
                    instruction = Instruction(
                        MoveCall(move_from=move_from.id, move_to=move_to.id)
                    )
                    self.chain.instructions.append(instruction)

    def call_function(  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _EmitterValue:
        """Performs actions for the call_function node of a graph module.

        Dispatches based on 'target' and emits the corresponding function. 'call_function' is a
        powerful node that contains many operations ranging from control_flow, to memory management,
        to delegate and operator calls.
        """

        # Delegate and operator calls are the only functions that should have a debug handle
        # associated with them. All the others such as memory.alloc, getitem should be ignored.
        # Default to none and let delegates and ops override.
        if target == operator.getitem:
            assert len(args) == 2
            head = typing.cast(Mapping[int, _EmitterValue], args[0])
            index = typing.cast(int, args[1])
            return head[index]

        elif target == memory.alloc:
            assert len(args) == 1
            return self._emit_spec(self.node.meta["spec"])

        elif target == memory.view:
            return self._emit_view(args)

        elif target == memory.free:
            assert len(args) == 1
            # pyre-ignore
            return self._emit_free(args[0])

        elif target is torch.ops.higher_order.cond:
            return self._emit_control_flow(target, args, kwargs)

        elif target is torch.ops.higher_order.map_impl:
            return self._emit_control_flow(target, args, kwargs)

        elif target == executorch_call_delegate:
            lowered_module = args[0]
            assert is_lowered_module(lowered_module)
            v = self._emit_delegate(lowered_module, args[1:], kwargs)
            delegate_instruction_id = len(self.chain.instructions) - 1
            self._add_debug_handle(delegate_instruction_id, target, lowered_module)
            self._add_delegate_map(lowered_module, delegate_instruction_id)
            return v

        elif isinstance(
            target, (torch._ops.OpOverload, EdgeOpOverload, BackendOpOverload)
        ):
            return self._emit_operator(target, args, kwargs)

        else:
            raise InternalError(
                self._emit_node_specific_error(
                    self.node, f"invalid target for call_function {target}"
                )
            )

    def run(  # pyre-fixme[14]
        self,
        *args: _Argument,
        initial_env: Optional[Dict[torch.fx.Node, _Argument]] = None,
    ) -> None:
        """Traverses all nodes in the graph module and emits each one appropriately."""
        super().run(*args, initial_env, enable_io_processing=False)

    def run_node(self, n: torch.fx.Node) -> None:
        """Executes and emits the specified node.

        For more context on what a node is and what execution means see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Node
        """
        self.node = n
        try:
            ret = super().run_node(n)
        except Exception as e:
            if isinstance(e, (InternalError, ExportError)):
                raise e
            else:
                raise InternalError(
                    self._emit_node_specific_error(self.node, str(e))
                ) from e
        return ret


class _TopLevelEmitter(_Emitter):
    """An emitter that manages the root level operations within a graph module.

    Exists as a separate class so that 'Emitter' can handle the special behavior of 'placeholder'
    and 'output' nodes in control flow submodules.
    """

    def __init__(
        self,
        name: str,
        exported_program: ExportedProgram,
        graph_module: torch.fx.GraphModule,
        program_state: _ProgramState,
        emitter_state: _EmitterState,
    ) -> None:
        super().__init__(graph_module, emitter_state, program_state)
        self.name = name
        self.exported_program = exported_program

        self.inputs: List[int] = []
        self.outputs: List[int] = []
        self.given_mutable_buffer_warning = False

        def create_container_str(spec: Optional[pytree.TreeSpec]) -> str:
            if spec is None:
                return ""
            assert isinstance(spec, pytree.TreeSpec), type(spec)
            dummy_leaves = [0] * spec.num_leaves
            tree = torch.utils._pytree.tree_unflatten(dummy_leaves, spec)
            # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
            _, tree = ex_pytree.tree_flatten(tree)
            return tree.to_str()

        inp_container_str = create_container_str(exported_program.call_spec.in_spec)
        out_container_str = create_container_str(exported_program.call_spec.out_spec)

        self.container_meta_type = ContainerMetadata(
            inp_container_str, out_container_str
        )

    def _find_fqn_for_placeholder(
        self, target: _Target, spec: Any  # pyre-ignore[2]
    ) -> Tuple[Optional[str], bool]:
        # Find the fully qualified name
        fqn = None
        is_mutable_buffer = False
        if target in self.exported_program.graph_signature.inputs_to_parameters:
            fqn = self.exported_program.graph_signature.inputs_to_parameters[target]

        elif target in self.exported_program.graph_signature.inputs_to_buffers:
            fqn = self.exported_program.graph_signature.inputs_to_buffers[target]

            # if the buffer is mutated then record that
            if fqn in self.exported_program.graph_signature.buffers_to_mutate.values():
                is_mutable_buffer = True
                if not self.given_mutable_buffer_warning:
                    warnings.warn(
                        "Mutation on a buffer in the model is detected. ExecuTorch assumes "
                        "buffers that are mutated in the graph have a meaningless initial state, "
                        "only the shape and dtype will be serialized.",
                        UserWarning,
                        stacklevel=1,
                    )
                    self.given_mutable_buffer_warning = True

        elif (
            target
            in self.exported_program.graph_signature.inputs_to_lifted_tensor_constants
        ):
            fqn = (
                self.exported_program.graph_signature.inputs_to_lifted_tensor_constants[
                    target
                ]
            )
        return fqn, is_mutable_buffer

    def placeholder(
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> _AbstractValue:
        """Emits the value within the placeholder node.

        For more information on placeholder nodes see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.placeholder
        """
        spec = self.node.meta["spec"]
        is_user_input = True

        if isinstance(target, str) and isinstance(spec, TensorSpec):
            fqn, is_mutable_buffer = self._find_fqn_for_placeholder(target, spec)

            # From the fqn find the corresponding tensor
            real_tensor = None
            if fqn in self.exported_program.state_dict:
                real_tensor = self.exported_program.state_dict[fqn]
                is_user_input = False

            elif fqn in self.exported_program.constants:
                real_tensor = self.exported_program.constants[fqn]
                is_user_input = False
            elif fqn is not None:
                buffers = self.exported_program.named_buffers()
                buf = next((x[1] for x in buffers if x[0] == fqn), None)
                if buf is not None:
                    real_tensor = buf
                    is_user_input = False
                else:
                    raise InternalError(
                        self._emit_node_specific_error(
                            self.node,
                            f"Could not find buffer with fqn {fqn} in state_dict or named_buffers",
                        )
                    )

            # assign the storage of the placeholder spec to the storage of the real tensor if there is one
            if real_tensor is not None:
                # for non-contigous tensors, convert to a contiguous one
                real_tensor = real_tensor.contiguous()
                # Weights cannot be views during emission or serialization
                if real_tensor.nbytes != real_tensor.untyped_storage().nbytes():
                    real_tensor = real_tensor.clone()

                spec.storage = real_tensor.untyped_storage()

            # User inputs and mutable buffers are not constants, other buffers or parameters are.
            spec.const = not (is_user_input or is_mutable_buffer)

        evalue = (
            self._tensor_spec_to_evalue(spec)
            if isinstance(spec, TensorSpec)
            else self._constant_to_evalue(spec, None)
        )
        value = self._emit_evalue(evalue)

        # Only user inputs should remain as inputs.
        if is_user_input:
            self.inputs.append(value.id)

        return value

    def output(
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> None:
        """Records the ExecutionPlan's outputs based on the output node in the graph."""
        if isinstance(args[0], dict):
            args_tuple, _ = pytree.tree_flatten(args[0])
        else:
            args_tuple = typing.cast(Tuple[_AbstractValue, ...], args[0])
        if isinstance(args_tuple, _AbstractValue):
            self.outputs.append(args_tuple.id)
        else:
            for arg in args_tuple:
                if isinstance(arg, (int, float, bool, type(None))):
                    arg = self._emit_evalue(self._constant_to_evalue(arg, None))
                elif isinstance(arg, str):
                    # TODO(jackkhuu): T181599879 Add support for string outputs IFF compiler supports
                    raise InternalError(
                        self._emit_node_specific_error(
                            self.node,
                            f"Returning {arg} is not yet supported in the emitter.",
                        )
                    )
                else:
                    # Every other output should already have its value emitted.
                    # They should only be abstract IDs at this point.
                    assert isinstance(arg, _AbstractValue)
                self.outputs.append(arg.id)

    def plan(self) -> ExecutionPlan:
        """Returns the execution plan emitted from this entry point."""
        return ExecutionPlan(
            name=self.name,
            values=self.emitter_state.values,
            inputs=self.inputs,
            outputs=self.outputs,
            chains=[self.chain],
            operators=self.emitter_state.operators,
            delegates=self.emitter_state.delegates,
            # non_const_buffer_sizes field is set by the memory_planning_pass. In case the field is
            # missing in scenarios like unit test that does not enable memory planning, assume an
            # empty list.
            non_const_buffer_sizes=typing.cast(
                List[int], self.module.meta["non_const_buffer_sizes"]
            ),
            container_meta_type=self.container_meta_type,
        )
