# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch._export
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.error import ExportError
from executorch.exir.pass_manager import PassType
from executorch.exir.passes import (
    aten_to_edge_passes,
    EdgeToBackendOpsPass,
    OpReplacePass,
    SymShapeEvalPass,
)
from executorch.exir.passes.remove_assert_async_pass import RemoveAssertAsyncPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.schema import Program
from executorch.exir.serialize import serialize_to_flatbuffer
from executorch.exir.verification.verifier import (
    EXIRATenDialectVerifier,
    EXIREdgeDialectVerifier,
)
from torch._export import ExportedProgram
from torch.fx._compatibility import compatibility

Val = Any


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
            self._buffer = serialize_to_flatbuffer(
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

    op_replace_pass = [OpReplacePass()] if config._use_edge_ops else []
    passes = aten_to_edge_passes.passes + op_replace_pass + config.passes
    new_ep = copy.deepcopy(ep).transform(*passes)
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
        SymShapeEvalPass(),
        config.to_out_var_pass,
        config.memory_planning_pass,
    ]
    return passes


######## MULTI METHOD STUFF BELOW HERE. TO BE MERGED INTO ExirExportedProgram and ExecutorchProgram AND THEN DELETED ##########


# MultiMethodExirExportedProgram represents an exported program that contains
# multiple methods, all as valid entry points to the program.
#
# Internally, each method is represented as a separate ExirExportedProgram.
# Methods (fx.GraphModule's) do not share anything with each other to
# ensure that each is self-contained. This is important because transformation
# passes can be local and do not need to concern themselves about other methods
# that exists on the same MultiMethodExirExportedProgram.
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
            self._buffer = serialize_to_flatbuffer(
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

    # TODO(ycao): This doesn't make sense any more, remove/change later.
    def dump_graph_module(self) -> torch.fx.GraphModule:
        return self.get_multi_method_graph_module().module

    def get_multi_method_graph_module(self) -> "MultiMethodExirExportedProgram":
        return self._executorch_dialect_ir_program


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
