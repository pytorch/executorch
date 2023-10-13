# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Type, Union

import torch
import torch._export
from executorch.exir._serialize import _serialize_pte_binary
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.partitioner import TPartitioner
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.emit._emitter import _DelegateDebugIdentifierMap
from executorch.exir.error import ExportError
from executorch.exir.pass_manager import PassType
from executorch.exir.passes import (
    aten_to_edge_passes,
    EdgeToBackendOpsPass,
    OpReplacePass,
)
from executorch.exir.passes.remove_assert_async_pass import RemoveAssertAsyncPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.print_program import pretty_print, print_program
from executorch.exir.schema import Program
from executorch.exir.tracer import _default_decomposition_table
from executorch.exir.verification.verifier import (
    EXIRATenDialectVerifier,
    EXIREdgeDialectVerifier,
)
from torch._export import ExportedProgram
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch._export.passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from torch.fx import _pytree as fx_pytree
from torch.fx._compatibility import compatibility
from torch.utils import _pytree as pytree

Val = Any


# Stub to ease migration from `transform` to private `_transform`
def transform_exported_program(ep, *passes: PassType) -> ExportedProgram:
    if hasattr(ep, "_transform"):
        return ep._transform(*passes)
    else:
        return ep.transform(*passes)


class HackedUpExportedProgramDONOTUSE(ExportedProgram):
    def __init__(
        self,
        root,
        graph,
        graph_signature,
        call_spec,
        state_dict,
        range_constraints,
        equality_constraints,
        module_call_graph,
        example_inputs,
    ):
        super().__init__(
            root,
            graph,
            graph_signature,
            call_spec,
            state_dict,
            range_constraints,
            equality_constraints,
            module_call_graph,
            example_inputs,
        )
        self._dialect = "HACKED_ATEN"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch._export.error as error

        if self.call_spec.in_spec is not None:
            user_args = args
            try:
                args = fx_pytree.tree_flatten_spec(user_args, self.call_spec.in_spec)  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(user_args)
                raise error.InternalError(
                    "Trying to flatten user inputs with exported input tree spec: \n"
                    f"{self.call_spec.in_spec}\n"
                    "but actually got inputs with tree spec of: \n"
                    f"{received_spec}"
                )

        ordered_params = tuple(
            self.state_dict[name] for name in self.graph_signature.parameters
        )
        ordered_buffers = tuple(
            self.state_dict[name] for name in self.graph_signature.buffers
        )

        with torch.no_grad():
            # NOTE: calling convention is first params, then buffers, then args as user supplied them.
            # See: torch/_functorch/aot_autograd.py#L1034
            res = torch.fx.Interpreter(self.graph_module).run(
                *ordered_params, *ordered_buffers, *args, enable_io_processing=False
            )

        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            mutated_buffers = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = list(assertion_dep_token.keys())[0]
                res = res[:assertion_dep_token_index]

            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
            finally:
                ix = 0
                for buffer in self.graph_signature.buffers_to_mutate.values():
                    self.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res


@compatibility(is_backward_compatible=False)
class ExirExportedProgram:
    def __init__(
        self,
        exported_program: ExportedProgram,
        after_to_edge_passes: bool,
    ):
        self.exported_program = exported_program

        # Add a flag to denote whehter to_edge is called on this program
        # to detect misusage of directly calling to_executorch without to_edge
        self.after_to_edge_passes = after_to_edge_passes

    def transform(self, *passes: PassType) -> "ExirExportedProgram":
        self.exported_program = self.exported_program._transform(*passes)
        return self

    def __call__(self, *args: Any) -> Any:
        return self.exported_program(*args)

    # TODO(ycao): Change this to a composable function.
    def to_edge(
        self, config: Optional[EdgeCompileConfig] = None
    ) -> "ExirExportedProgram":
        config = config or EdgeCompileConfig()
        assert isinstance(
            self.exported_program.graph_module, torch.fx.GraphModule
        ), f"type is instead: {type(self.exported_program.graph_module).__name__}"

        return _to_edge(self, config)

    def dump(self) -> None:
        print(self.exported_program.graph_module.graph)

    def to_executorch(
        self,
        config: Optional[ExecutorchBackendConfig] = None,
    ) -> "ExecutorchProgram":
        if not self.after_to_edge_passes:
            raise RuntimeError("Must run to_edge before to_executorch.")
        config = config or ExecutorchBackendConfig()
        ep = self.exported_program
        new_prog = ep._transform(*edge_to_executorch_passes(config))
        new_prog = ExirExportedProgram(new_prog, self.after_to_edge_passes)
        executorch_prog = ExecutorchProgram(
            new_prog,
            emit_stacktrace=config.emit_stacktrace,
            extract_segments=config.extract_segments,
            segment_alignment=config.segment_alignment,
            constant_tensor_alignment=config.constant_tensor_alignment,
            delegate_alignment=config.delegate_alignment,
        )
        executorch_prog.graph_module.meta.update(
            new_prog.exported_program.graph_module.meta
        )
        executorch_prog.graph_module.meta.update(
            self.exported_program.graph_module.meta
        )
        return executorch_prog

    def __deepcopy__(
        self, memo: Optional[Dict[int, Any]] = None
    ) -> "ExirExportedProgram":

        new_eep = ExirExportedProgram(
            copy.deepcopy(self.exported_program, memo),
            self.after_to_edge_passes,
        )
        return new_eep


@compatibility(is_backward_compatible=False)
class ExecutorchProgram:
    def __init__(
        self,
        exir_exported_program: ExirExportedProgram,
        emit_stacktrace: bool,
        extract_segments: bool,
        segment_alignment: int,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
    ) -> None:
        if not exir_exported_program.after_to_edge_passes:
            raise RuntimeError(
                "Need to call prog.to_edge prior to constructing ExecutorchProgram."
            )
        self.exported_program = exir_exported_program.exported_program
        self._buffer: Optional[bytes] = None
        self._emitter_output: Optional[EmitterOutput] = None
        self._emit_stacktrace: bool = emit_stacktrace
        self._extract_segments: bool = extract_segments
        self._segment_alignment: int = segment_alignment
        self._constant_tensor_alignment: Optional[int] = constant_tensor_alignment
        self._delegate_alignment: Optional[int] = delegate_alignment

    @property
    def buffer(self) -> bytes:
        if self._buffer is None:
            self._buffer = _serialize_pte_binary(
                program=self.program,
                extract_segments=self._extract_segments,
                segment_alignment=self._segment_alignment,
                constant_tensor_alignment=self._constant_tensor_alignment,
                delegate_alignment=self._delegate_alignment,
            )
        return self._buffer

    @property
    def program(self) -> Program:
        if self._emitter_output is None:
            self._emitter_output = emit_program(
                self.exported_program, self._emit_stacktrace
            )
        return self._emitter_output.program

    @property
    def debug_handle_map(self) -> Dict[int, Union[int, List[int]]]:
        if self._emitter_output:
            return self._emitter_output.debug_handle_map
        return {}

    @property
    def delegate_map(
        self,
    ) -> Dict[str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]]:
        if self._emitter_output:
            return self._emitter_output.method_to_delegate_debug_id_map
        return {}

    @property
    def graph_module(self) -> torch.fx.GraphModule:
        return self.exported_program.graph_module

    # TODO (zhxchen17) Change this to property.
    def dump_graph_module(self) -> torch.fx.GraphModule:
        return self.exported_program.graph_module

    def dump_exported_program(self) -> ExportedProgram:
        return self.exported_program


def _to_edge(ep, config: EdgeCompileConfig) -> "ExirExportedProgram":
    if config._check_ir_validity:
        try:
            EXIRATenDialectVerifier()(ep.exported_program.graph_module)
        except ExportError:
            logging.info(
                "If you'd like to disable IR validation checking, please set _check_ir_validity in EdgeCompileConfig, "
                "like *.to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))."
            )
            raise

    # TODO: the last two passes for aten_to_edge need to be eliminated_dead_code -> debug_handle_generator. After enable
    # use_edge_op it can be moved to aten_to_edge_passes before eliminated_dead_code pass. Also ExportPass doesn't play
    # well with node.meta, meaning after some passes permuting operators, we may lose some information in node.meta.
    # It might be regenerated in SpecPropPass so it may not be visiable. However debug handle will be lost.
    pre_op_replace_passes = aten_to_edge_passes.passes[:-2]
    post_op_replace_passes = aten_to_edge_passes.passes[-2:]

    new_ep = copy.deepcopy(ep).transform(*pre_op_replace_passes)
    if new_ep.exported_program.dialect == "ATEN":
        new_ep.exported_program = lift_constant_tensor_pass(new_ep.exported_program)

    if config._use_edge_ops:
        new_ep = new_ep.transform(OpReplacePass())

    new_ep = new_ep.transform(*post_op_replace_passes)
    new_ep.exported_program = ExportedProgram(
        new_ep.exported_program.graph_module,
        new_ep.exported_program.graph,
        new_ep.exported_program.graph_signature,
        new_ep.exported_program.call_spec,
        new_ep.exported_program.state_dict,
        new_ep.exported_program.range_constraints,
        new_ep.exported_program.equality_constraints,
        new_ep.exported_program.module_call_graph,
        new_ep.exported_program.example_inputs,
        dialect="EDGE",
    )
    if config._check_ir_validity:
        EXIREdgeDialectVerifier(check_edge_ops=config._use_edge_ops)(
            new_ep.exported_program.graph_module
        )
    new_ep.after_to_edge_passes = True
    return new_ep


def edge_to_executorch_passes(config: ExecutorchBackendConfig) -> List[PassType]:
    # pyre-ignore
    passes: List[PassType] = [
        *config.passes,
        SpecPropPass(),
        EdgeToBackendOpsPass(),
        RemoveAssertAsyncPass(),
        config.sym_shape_eval_pass,
        config.to_out_var_pass,
        config.memory_planning_pass,
    ]
    return passes


# MultiMethodExirExportedProgram represents an exported program that contains
# multiple methods, all as valid entry points to the program.
#
# Internally, each method is represented as a separate ExirExportedProgram.
# Methods (fx.GraphModule's) do not share anything with each other to
# ensure that each is self-contained. This is important because transformation
# passes can be local and do not need to concern themselves about other methods
# that exists on the same MultiMethodExirExportedProgram.
#
# TODO(T152006915): Merge this into ExirExportedProgram and then delete it.
@compatibility(is_backward_compatible=False)
class MultiMethodExirExportedProgram:
    def __init__(
        self,
        progs: Dict[str, ExirExportedProgram],
        getters: Optional[Dict[str, Any]] = None,
    ):
        # TODO(ycao): Support merging use case where user started by creating
        # an empty MultiMethodExirExportedProgram and then start adding more
        # graph modules to it.
        assert (
            len(progs) > 0
        ), "Expected at least 1 graph module in MultiMethodExirExportedProgram"
        self._method_to_program = progs
        self._method_to_prim_getter = getters

    # Get the default method, which is either the only method contained
    # in this MultiMethodExirExportedProgram or the method named `forward`.
    def _get_default_program(self):
        if len(self._method_to_program) == 1:
            return next(iter(self._method_to_program.values()))
        elif "forward" in self._method_to_program:
            return self._method_to_program["forward"]
        else:
            return None

    def save(self) -> None:
        # TODO(ycao): Implement.
        raise NotImplementedError()

    def load(self) -> None:
        # TODO(ycao): Implement.
        raise NotImplementedError()

    def find_method(self, name: str) -> Optional[ExirExportedProgram]:
        return self._method_to_program.get(name)

    def merge(self, other: "MultiMethodExirExportedProgram"):
        for method_name, program in other.methods().items():
            assert (
                method_name not in self._method_to_program
            ), f"There already is a method named {method_name} in this program"
            self._method_to_program[method_name] = program

    def transform(self, *passes: PassType) -> "MultiMethodExirExportedProgram":
        method_name_to_transformed_program = {
            method_name: prog.transform(*passes)
            for method_name, prog in self._method_to_program.items()
        }
        return MultiMethodExirExportedProgram(method_name_to_transformed_program)

    def methods(self) -> Dict[str, ExirExportedProgram]:
        return self._method_to_program

    def prim_getters(self) -> Optional[Dict[str, Any]]:
        return self._method_to_prim_getter

    def __call__(self, *args: Val, **kwargs: Val) -> Val:
        prog = self._get_default_program()

        assert (
            prog is not None
        ), """MultiMethodExirExportedProgram can not be called directly unless "
        "it only contains a single method or it contains a `forward` method. "
        "Please look up one of its methods first via "
        "`MultiMethodExirExportedProgram.find_method(method_name)`."""

        return prog(*args, **kwargs)

    def __repr__(self) -> str:
        # TODO(ycao): Implement.
        raise NotImplementedError()

    def __str__(self) -> str:
        # TODO(ycao): Implement a real one.
        return super().__str__()

    def access_property_of_default_method(self, property_name: str):
        default_program = self._get_default_program()
        assert (
            default_program is not None
        ), f"""Exported program contains more than one methods and none of them "
        "is named `forward`, it is impossible to identify the default method. "
        "please look up one of its methods first via `find_method(method_name)` "
        "to access property: {property_name}."""
        return getattr(default_program.exported_program.graph_module, property_name)

    @property
    def graph(self):
        return self.access_property_of_default_method("graph")

    @property
    def code(self):
        return self.access_property_of_default_method("code")

    @property
    def module(self):
        default_prog = self._get_default_program()
        assert (
            default_prog is not None
        ), """Exported program contains more than"
        " one methods and none of them is named `forward`,"
        " it is impossible to identify the default method "
        "to fetch GraphModule for."""
        return default_prog.exported_program.graph_module

    # TODO(ycao): Implement custom __reduce__ to account for lost of
    # meta['val']

    # TODO(ycao): Change this to a composable function.
    def to_edge(
        self, config: Optional[EdgeCompileConfig] = None
    ) -> "MultiMethodExirExportedProgram":
        if config is None:
            config = EdgeCompileConfig()
        method_name_to_edge_prog = {
            method_name: prog.to_edge(config)
            for method_name, prog in self.methods().items()
        }
        return MultiMethodExirExportedProgram(
            method_name_to_edge_prog,
            self.prim_getters(),
        )

    # TODO(ycao): Change this to a composable function.
    def to_executorch(
        self,
        config: Optional[ExecutorchBackendConfig] = None,
    ) -> "MultiMethodExecutorchProgram":
        return multi_method_program_to_executorch(self, config)


# TODO(T152006915): Merge this into ExecutorchProgram and then delete it.
@compatibility(is_backward_compatible=False)
class MultiMethodExecutorchProgram:
    def __init__(
        self,
        executorch_dialect_program: "MultiMethodExirExportedProgram",
        emit_stacktrace: bool,
        extract_segments: bool,
        segment_alignment: int,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
        prim_getters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._buffer: Optional[bytes] = None
        temp: Dict[str, ExportedProgram] = {}
        for name, prog in executorch_dialect_program.methods().items():
            temp[name] = prog.exported_program
        self._emitter_output: EmitterOutput = emit_program(
            temp,
            emit_stacktrace,
            executorch_dialect_program.prim_getters(),
        )
        self._executorch_dialect_ir_program = executorch_dialect_program
        self._extract_segments: bool = extract_segments
        self._segment_alignment: int = segment_alignment
        self._constant_tensor_alignment: Optional[int] = constant_tensor_alignment
        self._delegate_alignment: Optional[int] = delegate_alignment
        self._prim_getter_cache = prim_getters

    @property
    def buffer(self) -> bytes:
        if self._buffer is None:
            self._buffer = _serialize_pte_binary(
                program=self._emitter_output.program,
                extract_segments=self._extract_segments,
                segment_alignment=self._segment_alignment,
                constant_tensor_alignment=self._constant_tensor_alignment,
                delegate_alignment=self._delegate_alignment,
            )
        return self._buffer

    @property
    def program(self) -> Program:
        return self._emitter_output.program

    @property
    def debug_handle_map(self) -> Dict[int, Union[int, List[int]]]:
        return self._emitter_output.debug_handle_map

    @property
    def delegate_map(
        self,
    ) -> Dict[str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]]:
        if self._emitter_output:
            return self._emitter_output.method_to_delegate_debug_id_map
        return {}

    # TODO(ycao): This doesn't make sense any more, remove/change later.
    def dump_graph_module(self) -> torch.fx.GraphModule:
        return self.get_multi_method_graph_module().module

    def get_multi_method_graph_module(self) -> "MultiMethodExirExportedProgram":
        return self._executorch_dialect_ir_program


# TODO(T152006915): Merge this into to_executorch and then delete it.
def multi_method_program_to_executorch(
    edge_dialect_program: MultiMethodExirExportedProgram,
    config: Optional[ExecutorchBackendConfig] = None,
) -> MultiMethodExecutorchProgram:
    config = config or ExecutorchBackendConfig()
    passes = edge_to_executorch_passes(config)
    return MultiMethodExecutorchProgram(
        executorch_dialect_program=edge_dialect_program.transform(*passes),
        emit_stacktrace=config.emit_stacktrace,
        extract_segments=config.extract_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
        prim_getters=edge_dialect_program.prim_getters(),
    )


def to_edge(
    programs: Union[ExportedProgram, Dict[str, ExportedProgram]],
    constant_methods: Optional[Dict[str, Any]] = None,
    compile_config: Optional[EdgeCompileConfig] = None,
) -> "EdgeProgramManager":
    """
    :func:`to_edge` constructs an EdgeProgramManger from a set of exported programs in
    ATen dialect. Upon construction those programs are transformed into edge dialect.

    Args:
        programs: Can be a single ExportedProgram or a dictionary mapping function names to their corresponding ExportedPrograms. If only a single ExportedProgram is provided it will be assigned the name "forward".

        constant_methods: An optional dictionary of method name to the constant value returned by that method in eager mode. Often used to store config information on Edge models.

        compile_config: An optional argument used to provide greater control over the transformation to edge dialect process.

    Returns:
        EdgeProgramManager
    """
    assert not isinstance(constant_methods, EdgeCompileConfig)
    config = compile_config or EdgeCompileConfig()
    if not isinstance(programs, dict):
        aten_programs = {"forward": programs}
    else:
        aten_programs = programs

    edge_programs: Dict[str, ExportedProgram] = {}
    for name, program in aten_programs.items():
        # Decompose to Core ATen
        program = program.run_decompositions(
            _default_decomposition_table()  # pyre-ignore[6]
        )

        if config._check_ir_validity:
            try:
                EXIRATenDialectVerifier()(program.graph_module)
            except ExportError as e:
                logging.info(f"Input program {name} is not in ATen dialect.")
                raise e

        op_replace_pass = [OpReplacePass()] if config._use_edge_ops else []

        # TODO: the last two passes for aten_to_edge need to be eliminated_dead_code -> debug_handle_generator. After enable
        # use_edge_op it can be moved to aten_to_edge_passes before eliminated_dead_code pass. Also ExportPass doesn't play
        # well with node.meta, meaning after some passes permuting operators, we may lose some information in node.meta.
        # It might be regenerated in SpecPropPass so it may not be visiable. However debug handle will be lost.
        program = lift_constant_tensor_pass(program)
        passes = []
        passes.append(
            ReplaceViewOpsWithViewCopyOpsPass()
        )  # TODO move inside aten_to_edge passes after all users are migrated off v1 capture
        passes.extend(aten_to_edge_passes.passes[:-2])
        passes.extend(op_replace_pass)
        passes.extend(aten_to_edge_passes.passes[-2:])
        edge_program = program._transform(*passes)
        if config._check_ir_validity:
            try:
                EXIREdgeDialectVerifier(check_edge_ops=config._use_edge_ops)(
                    edge_program.graph_module
                )
            except ExportError as e:
                logging.info(f"Resultant program {name} is not in edge dialect.")
                raise e
        edge_programs[name] = edge_program
    return EdgeProgramManager(edge_programs, constant_methods)


class EdgeProgramManager:
    """
    Package of one or more `ExportedPrograms` in Edge dialect. Designed to simplify
    lowering to ExecuTorch.

    Allows easy applications of transforms across a collection of exported programs
    including the delegation of subgraphs.

    Manages the second link in the lowering chain of ATen -> Edge -> ExecuTorch.
    """

    # TODO(T163717152): Link to Edge dialect docs here ^.

    def __init__(
        self,
        edge_programs: Dict[str, ExportedProgram],
        constant_methods: Optional[Dict[str, Any]] = None,
    ):
        """
        Should not be called directly by users. User should use :func:'to_edge' instead.

        Constructs an EdgeProgramManager from an existing set of exported programs in edge dialect.
        """

        for name, program in edge_programs.items():
            try:
                EXIREdgeDialectVerifier()(program.graph_module)
            except ExportError as e:
                logging.info(f"Input program {name} is not in aten dialect.")
                raise e

        self._edge_programs = edge_programs
        self._config_methods = constant_methods

    @property
    def methods(self) -> Set[str]:
        """
        Returns the set of methods in this EdgeProgramManager.
        """
        return set(self._edge_programs.keys())

    @property
    def config_methods(self) -> Set[str]:
        """
        Returns the set of config methods in this EdgeProgramManager.
        """
        return set(self._config_methods.keys()) if self._config_methods else set()

    def exported_program(self, method_name: str = "forward") -> ExportedProgram:
        """
        Returns the ExportedProgram specified by 'method_name'.
        """
        return self._edge_programs[method_name]

    def transform(
        self,
        passes: Union[Sequence[PassType], Dict[str, Sequence[PassType]]],
    ) -> "EdgeProgramManager":
        """
        Transforms the program according to the provided passes.

        Args:
            passes: The passes can either be a list of passes, or a
                dictionary mapping method names to lists of passes. If it is
                just a list of passes, all methods in the given EdgeProgramManager
                will be transformed with the provided passes. If it is a
                dictionary, only method names specified in the dictionary will be
                transformed with their corresponding passes.

        Returns:
            EdgeProgramManager: A copy of the calling EdgeProgramManager with the
            transformations applied.
        """
        new_programs: Dict[str, ExportedProgram] = {}
        if isinstance(passes, dict):
            for name, program in self._edge_programs.items():
                if name in passes.keys():
                    new_programs[name] = program._transform(*passes[name])
                    EXIREdgeDialectVerifier()(new_programs[name].graph_module)
                else:
                    new_programs[name] = copy.deepcopy(program)

        else:  # apply passes to every method
            for name, program in self._edge_programs.items():
                new_programs[name] = program._transform(*passes)
                EXIREdgeDialectVerifier()(new_programs[name].graph_module)

        return EdgeProgramManager(
            new_programs,
            copy.deepcopy(self._config_methods),
        )

    def to_backend(
        self, partitioner: Union[Type[TPartitioner], Dict[str, Type[TPartitioner]]]
    ) -> "EdgeProgramManager":
        """
        Returns a semantically-equivalent program to the one given as input,
        but with portions of each program in the EdgeProgramManager targeted
        for delegation as determined by the partitioner.

        Args:
            partitioner: The partitioner can either be a Partitioner subclass, or a
                dictionary mapping method names to Partitioner subclass. If it is a
                Partitioner subclass, all programs in the given EdgeProgramManager
                will be lowered using the given partitioner. If it is a
                dictionary, only method names specified in the dictionary will be
                lowered with the given partitioner.

                The Partitioner subclass is in charge with tagging portions of the
                input program for delegation. A valid partitioner must have
                partition_tags: Dict[str, DelegationSpec], where each key is a tag
                name and the nodes with same tag will be fused a one subgraph and
                delegated to backend specififed in delegation spec.

        Returns:
            EdgeProgramManager: A copy of the calling EdgeProgramManager with the
            specified subgraphs lowered.
        """
        new_edge_programs: Dict[str, ExportedProgram] = {}
        if isinstance(partitioner, dict):
            for name, program in self._edge_programs.items():
                if name in partitioner.keys():
                    new_edge_programs[name] = to_backend(program, partitioner[name])
                else:
                    new_edge_programs[name] = copy.deepcopy(program)

        else:  # apply partitioner to every method
            for name, program in self._edge_programs.items():
                new_edge_programs[name] = to_backend(program, partitioner)

        return EdgeProgramManager(
            new_edge_programs, copy.deepcopy(self._config_methods)
        )

    def to_executorch(
        self, config: Optional[ExecutorchBackendConfig] = None
    ) -> "ExecutorchProgramManager":
        """
        Transforms the program to the ExecuTorch backend.

        Args:
            config: An optional argument used to provide greater control over
                the transformation to the ExecuTorch backend.

        Returns:
            ExecutorchProgramManager: A manager representing the state of the EdgeProgramManager
            after it has been transformed to the ExecuTorch backend.
        """
        config = config if config else ExecutorchBackendConfig()

        execution_programs: Dict[str, ExportedProgram] = {}
        for name, program in self._edge_programs.items():
            new_prog = program._transform(*edge_to_executorch_passes(config))
            execution_programs[name] = new_prog

        return ExecutorchProgramManager(
            execution_programs, self._config_methods, config
        )


class ExecutorchProgramManager:
    """
    Package of one or more :class:'ExportedPrograms' in Execution dialect. Designed to simplify
    lowering to ExecuTorch.

    When the ExecutorchProgramManager is constructed the ExportedPrograms in execution dialect
    are used to form the executorch binary (in a process called emission) and then serialized
    to a buffer.

    Manages the final link in the lowering chain of ATen -> Edge -> ExecuTorch.
    """

    # TODO(T163717152): Link to Execution dialect docs here ^.

    def __init__(
        self,
        execution_programs: Dict[str, ExportedProgram],
        config_methods: Optional[Dict[str, Any]] = None,
        backend_config: Optional[ExecutorchBackendConfig] = None,
    ):
        """
        End users should not call this constructor directly. Instead, they should use
        :func:'to_executorch' to construct an ExecutorchProgramManger.

        Constructs an ExecutorchProgramManager from a set of exported programs in
        execution dialect.

        Args:
            execution_programs: A dictionary of method name to the corresponding
            ExportedProgram.

            config_methods: A dictionary of method name to the config value returned
            by that method in eager mode.

            backend_config: An optional argument used to provide greater control over
            the emission and serialization.
        """
        # Set up methods
        self._execution_programs: Dict[str, ExportedProgram] = execution_programs
        self._config_methods: Optional[Dict[str, Any]] = config_methods

        backend_config = backend_config or ExecutorchBackendConfig()

        # Emit methods
        self._emitter_output: EmitterOutput = emit_program(
            self._execution_programs,
            backend_config.emit_stacktrace,
            self._config_methods,
        )

        # Serialize emitter output to a buffer
        self._buffer: bytes = _serialize_pte_binary(
            program=self._emitter_output.program,
            extract_segments=backend_config.extract_segments,
            segment_alignment=backend_config.segment_alignment,
            constant_tensor_alignment=backend_config.constant_tensor_alignment,
            delegate_alignment=backend_config.delegate_alignment,
        )

    @property
    def methods(self) -> Set[str]:
        """
        Returns the set of methods in this ExecutorchProgramManager.
        """
        return set(self._execution_programs.keys())

    @property
    def config_methods(self) -> Set[str]:
        """
        Returns the set of config methods in this ExecutorchProgramManager.
        """
        return set(self._config_methods.keys()) if self._config_methods else set()

    def exported_program(self, method_name: str = "forward") -> ExportedProgram:
        """
        Returns the ExportedProgram specified by 'method_name'.
        """
        return self._execution_programs[method_name]

    def dump_executorch_program(self, verbose: bool = False) -> None:
        """
        Prints the ExecuTorch binary in a human readable format.

        Args:
            verbose (bool):
                If False prints the binary in a condensed format.
                If True prints the binary 1-1 with the specification in the schema.
        """
        if verbose:
            pretty_print(self._emitter_output.program)
        else:
            print_program(self._emitter_output.program)

    @property
    def debug_handle_map(self) -> Dict[int, Union[int, List[int]]]:
        return self._emitter_output.debug_handle_map

    @property
    def delegate_map(
        self,
    ) -> Dict[str, Dict[int, Dict[str, Union[str, _DelegateDebugIdentifierMap]]]]:
        return self._emitter_output.method_to_delegate_debug_id_map

    @property
    def executorch_program(self) -> Program:
        """
        Returns the object that represents the ExecuTorch binary before serialization.
        """
        return self._emitter_output.program

    @property
    def buffer(self) -> bytes:
        """
        Returns a buffer containing the serialized ExecuTorch binary.
        """
        return self._buffer
