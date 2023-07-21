import copy
import inspect
import logging
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import sympy
import torch
import torch._export
from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.error import ExportError, ExportErrorType, InternalError
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import (
    aten_to_edge_passes,
    EdgeToBackendOpsPass,
    MemoryPlanningPass,
    OpReplacePass,
    SymShapeEvalPass,
    ToOutVarPass,
)
from executorch.exir.passes.remove_assert_async_pass import RemoveAssertAsyncPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.schema import Program
from executorch.exir.serialize import serialize_to_flatbuffer
from executorch.exir.tracer import (
    _default_decomposition_table,
    dispatch_trace,
    dynamo_trace,
    ExirDynamoConfig,
    flatten_output,
    Value,
)
from executorch.exir.verification.verifier import (
    EXIRATenDialectVerifier,
    EXIREdgeDialectVerifier,
)
from functorch.experimental import functionalize
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.eval_frame import Constraint
from torch._export import CallSpec, export, ExportGraphSignature
from torch._export.exported_program import ExportedProgram
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    InputDim,
    RangeConstraint,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils import _pytree as pytree


Val = Any


def _unlift(gm, inp_pos_to_param_buffer_name, in_spec, out_spec, state_dict):
    count = 0
    # Step 1: make lifted params as get_attr
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if count in inp_pos_to_param_buffer_name:
                with gm.graph.inserting_after(node):
                    getattr_node = gm.graph.get_attr(
                        inp_pos_to_param_buffer_name[count]
                    )
                    node.replace_all_uses_with(getattr_node)
                    metadata = node.meta
                    gm.graph.erase_node(node)
                    getattr_node.meta = metadata
            count += 1

    # Step 2: Fix the input/output of the graph now that we deleted
    # some args.
    gm.graph.lint()
    names = [f"arg_{i}" for i in range(len(in_spec.children_specs))]
    gm.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )
    gm.recompile()

    # Step 3: Find state references in HigherOrderOps and recursively
    # fix them.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.cond:
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)
            inp_pos_to_param_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_param_buffer_name_for_submod[ix] = operand.target
                    true_gm.register_buffer(operand.target, state_dict[operand.target])
                    false_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (pred, true_graph, false_graph, real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                true_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
            )
            _unlift(
                false_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
            )
        if node.op == "call_function" and node.target.__name__ == "map_impl":
            body_graph, num_mapped, *operands = node.args
            body_gm = getattr(gm, body_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    body_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (body_graph, num_mapped, *real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                body_gm, inp_pos_to_buffer_name_for_submod, in_spec, None, state_dict
            )
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def unlift_exported_program_lifted_states(
    ep: torch._export.exported_program.ExportedProgram,
):
    new_gm = copy.deepcopy(ep.graph_module)

    # TODO Fix the period in params/buffers names later
    # maybe a pass to replace graph signature with fixed names
    param_buffer_name_to_corrected_name = {}

    for name, stuff in ep.state_dict.items():
        if name in ep.graph_signature.buffers:
            if "." in name:
                new_gm.register_buffer(name.replace(".", "_"), stuff)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_buffer(name, stuff)
        elif name in ep.graph_signature.parameters:
            if "." in name:
                new_gm.register_parameter(name.replace(".", "_"), stuff)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_parameter(name, stuff)
        else:
            raise AssertionError("encountered not registered param/buffer")

    count = 0
    inp_pos_to_param_buffer_name = {}
    for node in new_gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in ep.graph_signature.inputs_to_buffers:
                buffer_name = ep.graph_signature.inputs_to_buffers[node.name]
                if buffer_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[buffer_name]
                else:
                    inp_pos_to_param_buffer_name[count] = buffer_name
            if node.name in ep.graph_signature.inputs_to_parameters:
                param_name = ep.graph_signature.inputs_to_parameters[node.name]
                if param_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[param_name]
                else:
                    inp_pos_to_param_buffer_name[count] = param_name
            count += 1
    new_gm = _unlift(
        new_gm,
        inp_pos_to_param_buffer_name,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        ep.state_dict,
    )
    return new_gm


@compatibility(is_backward_compatible=False)
@dataclass
class CaptureConfig:
    pt2_mode: bool = True
    enable_functionalization: bool = True
    enable_dynamic_shape: bool = False
    enable_aot: bool = False
    _dynamo_config: "ExirDynamoConfig" = ExirDynamoConfig()
    _unlift: bool = False


@compatibility(is_backward_compatible=False)
@dataclass
class EdgeCompileConfig:
    passes: List[PassType] = field(default_factory=list)
    # TODO(qihan): remove ability to opt out
    _check_ir_validity: bool = True
    # TODO(larryliu): remove this
    _use_edge_ops: bool = False


@compatibility(is_backward_compatible=False)
@dataclass
class ServerCompileConfig:
    passes: List[PassType] = field(default_factory=list)


@compatibility(is_backward_compatible=False)
@dataclass
class ExecutorchBackendConfig:
    passes: List[PassType] = field(default_factory=list)
    memory_planning_pass: PassType = MemoryPlanningPass("greedy")
    to_out_var_pass: PassType = ToOutVarPass(ignore_to_out_var_failure=False)
    dynamic_memory_planning_mode: DynamicMemoryPlanningMode = (
        DynamicMemoryPlanningMode.UPPER_BOUND
    )
    emit_stacktrace: bool = False

    # Whether to move certain data blobs from the Program into separate
    # segments, rather than encoding those blobs in the flatbuffer data.
    # This makes it possible to free those blobs at runtime.
    extract_segments: bool = False

    # When extracting segments, the starting offset of each segment will be
    # aligned to this value (in bytes). When using mmap() to load segments, this
    # should be a multiple of the OS page size.
    segment_alignment: int = 4096

    # If provided, the minimum alignment of tensor buffers in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    constant_tensor_alignment: Optional[int] = None

    # If provided, the minimum alignment of delegate data in the program. Must
    # be a power of 2. If not provided, uses the value in the schema file.
    delegate_alignment: Optional[int] = None


# TODO(ycao): set up "__all__" to limit symbol exposure
def _to_edge(ep, config: EdgeCompileConfig) -> "ExirExportedProgram":
    if config._check_ir_validity:
        try:
            EXIRATenDialectVerifier()(ep.graph_module)
        except ExportError:
            logging.info(
                "If you'd like to disable IR validation checking, please set _check_ir_validity in EdgeCompileConfig, "
                "like *.to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))."
            )
            raise

    op_replace_pass = [OpReplacePass()] if config._use_edge_ops else []
    passes = aten_to_edge_passes.passes + op_replace_pass + config.passes
    new_ep = ep.transform(*passes)
    if config._check_ir_validity:
        EXIREdgeDialectVerifier(check_edge_ops=config._use_edge_ops)(
            new_ep.graph_module
        )
    new_ep.after_to_edge_passes = True
    return new_ep


@compatibility(is_backward_compatible=False)
class ExirExportedProgram(ExportedProgram):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        call_spec: CallSpec,
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]],
        range_constraints: Dict[sympy.Symbol, RangeConstraint],
        equality_constraints: List[Tuple[InputDim, InputDim]],
        after_to_edge_passes: bool,
    ):
        super().__init__(
            root,
            graph,
            graph_signature,
            call_spec,
            state_dict,
            range_constraints,
            equality_constraints,
        )

        # Add a flag to denote whehter to_edge is called on this program
        # to detect misusage of directly calling to_executorch without to_edge
        self.after_to_edge_passes = after_to_edge_passes

    def transform(self, *passes: PassType) -> "ExirExportedProgram":
        ep = super().transform(*passes)
        transformed_ep = ExirExportedProgram(
            ep.graph_module,
            ep.graph_module.graph,
            ep.graph_signature,
            ep.call_spec,
            ep.state_dict,
            ep.range_constraints,
            ep.equality_constraints,
            self.after_to_edge_passes,
        )
        transformed_ep.graph_module.meta.update(ep.graph_module.meta)
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        return transformed_ep

    # TODO(ycao): Change this to a composable function.
    def to_edge(
        self, config: Optional[EdgeCompileConfig] = None
    ) -> "ExirExportedProgram":
        config = config or EdgeCompileConfig()
        assert isinstance(
            self.graph_module, torch.fx.GraphModule
        ), f"type is instead: {type(self.graph_module).__name__}"

        return _to_edge(self, config)

    def dump(self) -> None:
        print(self.graph_module.graph)

    def _to_server(
        self, config: Optional[ServerCompileConfig] = None
    ) -> torch.nn.Module:
        config = config or ServerCompileConfig()
        res = PassManager(config.passes)(self.graph_module)
        assert res is not None
        # TODO ServerDialectGraphModule
        # return graph_module now.
        return res.graph_module

    @property
    def code(self):
        return self.graph_module.code

    def to_executorch(
        self,
        config: Optional[ExecutorchBackendConfig] = None,
    ) -> "ExecutorchProgram":
        if not self.after_to_edge_passes:
            raise RuntimeError("Must run to_edge before to_executorch.")
        config = config or ExecutorchBackendConfig()
        new_prog = self.transform(*edge_to_executorch_passes(config))
        executorch_prog = ExecutorchProgram(
            new_prog,
            emit_stacktrace=config.emit_stacktrace,
            extract_segments=config.extract_segments,
            segment_alignment=config.segment_alignment,
            constant_tensor_alignment=config.constant_tensor_alignment,
            delegate_alignment=config.delegate_alignment,
        )
        executorch_prog.graph_module.meta.update(new_prog.graph_module.meta)
        executorch_prog.graph_module.meta.update(self.graph_module.meta)
        return executorch_prog

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "ExportedProgram":
        gm = self.graph_module.__deepcopy__(memo)
        new_ep = ExirExportedProgram(
            gm,
            gm.graph,
            copy.deepcopy(self.graph_signature),
            copy.deepcopy(self.call_spec),
            copy.deepcopy(self.state_dict),
            copy.deepcopy(self.range_constraints),
            copy.deepcopy(self.equality_constraints),
            self.after_to_edge_passes,
        )
        new_ep.graph_module.meta.update(self.graph_module.meta)
        return new_ep

    def get_submodule(self, target: str):
        return self.graph_module.get_submodule(target)


@compatibility(is_backward_compatible=False)
class ExecutorchProgram:
    def __init__(
        self,
        exported_program: ExirExportedProgram,
        emit_stacktrace: bool,
        extract_segments: bool,
        segment_alignment: int,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
    ) -> None:
        if not exported_program.after_to_edge_passes:
            raise RuntimeError(
                "Need to call prog.to_edge prior to constructing ExecutorchProgram."
            )
        self.exported_program = exported_program
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
        return self.graph_module

    def dump_exported_program(self) -> ExirExportedProgram:
        return self.exported_program


# TODO(ycao): Move Executorch dialect to its own file
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
        self._emitter_output: EmitterOutput = emit_program(
            executorch_dialect_program.methods(),  # pyre-ignore
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


# This is to bootstrap the missing meta["val"] when 1. ph consists of scalar
# 2. meta["val"] is not properly set in dispatch_trace.
def _instantiate_missing_placeholder_val_with_real_inputs(gm, args):
    phs = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(phs) != len(args):
        raise ExportError(
            ExportErrorType.NOT_SUPPORTED,
            "Expect number of placeholders to be the same as user inputs.",
        )
    for node, arg in zip(phs, args):
        if "val" not in node.meta or node.meta["val"] is None:
            node.meta["val"] = arg


@compatibility(is_backward_compatible=False)
@torch.no_grad()
def capture(
    f: Callable[..., Any],
    args: Tuple[Value, ...],
    config: Optional[CaptureConfig] = None,
    constraints: Optional[List[Constraint]] = None,
) -> ExirExportedProgram:
    if not isinstance(args, tuple):
        raise ExportError(
            ExportErrorType.INVALID_INPUT_TYPE,
            f"Expect `args` to be a tuple, got type: {type(args)}.",
        )

    config = config or CaptureConfig()
    out_spec = None
    # TODO (zhxchen17) Always functionalize in a second pass no matter which path is taken.
    flat_args = tuple(pytree.tree_flatten(args)[0])
    if config.pt2_mode:
        if config.enable_aot:
            if not config.enable_functionalization:
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    "Functionalization is required for enable_aot.",
                )

            # TODO remove this later
            with patch("torch._export.DECOMP_TABLE", _default_decomposition_table()):
                ep = export(
                    f, args, _add_runtime_assertions=False, constraints=constraints
                )
            ep = ep.transform(ReplaceViewOpsWithViewCopyOpsPass())
            if not config._unlift:
                return ep  # pyre-ignore
            graph_module = unlift_exported_program_lifted_states(ep)

        elif config.enable_dynamic_shape:
            if not config._dynamo_config.dynamic_shapes:
                raise ExportError(
                    ExportErrorType.VIOLATION_OF_SPEC,
                    "Can't trace dynamo with static shapes under symbolic mode",
                )
            graph_module, _ = dynamo_trace(
                f,
                args,
                aten_graph=True,
                tracing_mode="symbolic",
                dynamo_config=config._dynamo_config,
                constraints=constraints,
            )

        else:
            graph_module, _ = dynamo_trace(
                f,
                args,
                aten_graph=True,
                tracing_mode="fake",
                dynamo_config=config._dynamo_config,
                constraints=None,  # constraints make sense only when dynamic shapes is enabled
            )

        if out_spec is None:
            out_spec = (
                graph_module.graph._codegen.pytree_info.out_spec
                or pytree.tree_flatten(f(*args))[1]
            )

        # NOTE (tmanlaibaatar)
        # torchdynamo.export adds extra kwarg into the graph module
        # which is then lost while we are calling make_fx. This is because
        # make_fx doesn't handle kwargs. Originally we used to use torchdynamo
        # input spec, but due to some limitations in pytree implementation, it doesn't
        # recognize the make_fx graph with torchdynamo input spec. We workaround it
        # by getting the input spec directly from user argument.
        in_spec = pytree.tree_flatten(args)[1]

        if config.enable_functionalization and not config.enable_aot:
            args = copy.deepcopy(args)

            def graph_with_interpreter(*args):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph_module).run(*args)

            functionalized_callable = functionalize(
                graph_with_interpreter,
                remove="mutations_and_views",
            )
            assert isinstance(functionalized_callable, Callable)

            if config.enable_dynamic_shape:
                fake_tensor_mode = FakeTensorMode(
                    allow_fallback_kernels=False,
                    allow_non_fake_inputs=True,
                    shape_env=ShapeEnv(),
                )

                inps: List[torch.Tensor] = []
                for node in graph_module.graph.nodes:
                    if node.op == "placeholder" and "val" in node.meta:
                        example_fake_tensor = node.meta["val"]
                        assert isinstance(example_fake_tensor, FakeTensor)
                        inps.append(example_fake_tensor)

                if detected_fake_mode := _guards.detect_fake_mode(inps):
                    fake_tensor_mode = detected_fake_mode

                count = 0

                def convert_to_fake(x):
                    nonlocal count
                    val = inps[count]
                    count += 1
                    return val

                fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)

                with enable_python_dispatcher(), fake_tensor_mode:
                    graph_module = make_fx(
                        functionalized_callable,
                        tracing_mode="real",
                        _allow_non_fake_inputs=True,
                    )(*fake_args)
            else:
                # To avoid breaking folks, use the deprecated "real" tracing
                # mode if we're not using pt2.
                tracing_mode = "fake" if config.pt2_mode else "real"
                graph_module = make_fx(
                    functionalized_callable,
                    tracing_mode=tracing_mode,
                    _allow_non_fake_inputs=True,
                )(*args)

        flatten_output(graph_module)

    else:
        warnings.warn(
            "exir.capture with pt2_mode=False is deprecated. Please use the default (pt2_mode=True) instead."
        )
        if not config.enable_functionalization:
            raise InternalError(
                "Can only disable functionalization under exir.capture() pt2 mode."
            )
        if config.enable_dynamic_shape:
            raise InternalError(
                "Can only enable dynamic shape tracing under exir.capture() pt2 mode."
            )
        if config.enable_aot:
            raise InternalError(
                "Using AOT mode is not supported for leagacy capture mode, please use pt2_mode=True instead."
            )
        graph_module = dispatch_trace(f, args)
        in_spec, out_spec = graph_module.in_spec, graph_module.out_spec

    _instantiate_missing_placeholder_val_with_real_inputs(graph_module, flat_args)
    graph_module._apply(torch.Tensor.contiguous)

    ep = ExirExportedProgram(
        graph_module,
        graph_module.graph,
        ExportGraphSignature([], [], [], [], {}, {}, {}, None),
        CallSpec(in_spec, out_spec),
        {},
        {},
        [],
        False,
    )
    return ep


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
        return getattr(default_program.graph_module, property_name)

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
        return default_prog.graph_module

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
    ) -> MultiMethodExecutorchProgram:
        return multi_method_program_to_executorch(self, config)


# Check that `method` is a valid method and raise an error otherwise.
def validate_is_method(method) -> None:
    assert inspect.ismethod(method), f"Expected a method, got {type(method)}"


CompileSpec = namedtuple(
    "CompileSpec", ["method_name", "callable", "args", "constraints"]
)


@compatibility(is_backward_compatible=False)
def capture_multiple(
    m: Union[torch.nn.Module, Callable[..., Any]],
    args: Union[Dict[str, Tuple[Value, ...]], Tuple[Value, ...]],
    config: Optional[CaptureConfig] = None,
    prim_getters: Optional[Set[str]] = None,
    constraints: Optional[Union[Dict[str, List[Constraint]], List[Constraint]]] = None,
):
    """
    capture_multiple traces either an nn.Module or just a callable with PyTorch
    operations inside and produce a single MultiMethodExirExportedProgram that
    can potentially have multiple entry points. When multiple entry points
    are traced, each of them is stored separately in the resulting
    MultiMethodExirExportedProgram without sharing state.

    Args:
        m: the `nn.Module` or callable to trace.

        args: Tracing example inputs.

        When `m` is an nn.Module, `args` can be
        1) A dictionary that maps names of method to their tracing example inputs.
        in this case, all specified methods will be captured.
        2) A tuple. In this case, `forward` method of `m` will be captured. It is
        equivalent to passing {"forward", tuple-type-args}

        When `m` is a non-Module callable, `args` must be a Tuple containing
        tracing example inputs.

        config: A CaptureConfig object that specifies how to interpret the
        program being captured.

        prim_getters: A set of primitive getter functions to capture the return values of

        constraints: Input shape constraints.

        When `m` is an nn.Module, `constraints` is a dictionary that maps names of method
        to their input shape constraints.

        When `m` is a non-Module callable, `constraints` is a list of input shape constraints.

    Returns:
        A MultiMethodExirExportedProgram.

        if `m` is an nn.Module, returned program would have multiple
        captured methods, each corresponding to one entry in args dictionary.

        if `m` is a non-Module callable, returned program would have a single
        captured method named `forward`.

    Raises:
        AssertionError if given method name do not reference a valid method
        on the given nn.Module.
    """
    # Normalize m and args.
    compile_specs = []
    prim_getter_cache: Optional[Dict[str, Any]] = None
    if isinstance(m, torch.nn.Module):
        if constraints is not None:
            assert isinstance(
                constraints, dict
            ), f"Expected a dict for constraints, got {type(constraints)}"

        if isinstance(args, tuple):
            compile_specs.append(
                CompileSpec(
                    "forward",
                    m.forward,
                    args,
                    constraints["forward"]
                    if constraints and "forward" in constraints
                    else None,
                )
            )
        else:
            assert isinstance(
                args, dict
            ), f"Expected a tuple or Dict[str, tuple], got {type(args)}"
            for method_name, method_args in args.items():
                compile_specs.append(
                    CompileSpec(
                        method_name,
                        getattr(m, method_name),
                        method_args,
                        constraints[method_name]
                        if constraints and method_name in constraints
                        else None,
                    )
                )
        if prim_getters is not None:
            prim_getter_cache = {}
            for getter in prim_getters:
                prim_getter_cache[getter] = getattr(m, getter)()
    else:
        # Reaching here means `m` is a non-Module callable.
        assert isinstance(
            m, Callable
        ), f"Only nn.Module or callable allowed, got {type(m)}"
        assert isinstance(
            args, tuple
        ), f"When tracing a non-Module callable, `args` must be a tuple of tracing inputs, but got {type(args)}"
        assert (
            prim_getters is None
        ), "Caller should not specify primitive getter functions when only providing a callable as input"
        if constraints is not None:
            assert isinstance(
                constraints, list
            ), f"Expected a list for constraints, got {type(constraints)}"
        compile_specs.append(CompileSpec("forward", m, args, constraints))

    method_name_to_prog = {}
    for compile_spec in compile_specs:
        method_name_to_prog[compile_spec.method_name] = capture(
            compile_spec.callable, compile_spec.args, config, compile_spec.constraints
        )

    return MultiMethodExirExportedProgram(method_name_to_prog, prim_getter_cache)


def edge_to_executorch_passes(config: ExecutorchBackendConfig) -> List[PassType]:
    # pyre-ignore
    passes: List[PassType] = [
        SpecPropPass(),
        *config.passes,
        EdgeToBackendOpsPass(),
        RemoveAssertAsyncPass(),
        SymShapeEvalPass(),
        config.to_out_var_pass,
        config.memory_planning_pass,
    ]
    return passes


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
