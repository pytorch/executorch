# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch
import torch._export

from executorch.exir._serialize import _serialize_pte_binary
from executorch.exir._serialize._cord import Cord
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.emit import emit_program, EmitterOutput
from executorch.exir.emit._emitter import _DelegateDebugIdentifierMap
from executorch.exir.error import ExportError
from executorch.exir.pass_manager import PassType
from executorch.exir.passes import (
    base_post_op_replace_passes,
    base_pre_op_replace_passes,
    dead_code_elimination_pass,
    EdgeToBackendOpsPass,
    MemoryFormatOpsPass,
    OpReplacePass,
)
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)
from executorch.exir.passes.normalize_view_copy_base_pass import (
    NormalizeViewCopyBasePass,
)
from executorch.exir.passes.remove_graph_asserts_pass import RemoveGraphAssertsPass
from executorch.exir.passes.remove_mixed_type_operators import RemoveMixedTypeOperators
from executorch.exir.passes.replace_view_copy_with_view_pass import (
    ReplaceViewCopyWithViewPass,
)
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.print_program import pretty_print, print_program
from executorch.exir.schema import Program
from executorch.exir.tracer import _default_decomposition_table
from executorch.exir.verification.verifier import (
    EXIRATenDialectVerifier,
    EXIREdgeDialectVerifier,
    get_aten_verifier,
)
from torch._export.passes import ReplaceViewOpsWithViewCopyOpsPass
from torch.export import ExportedProgram
from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputSpec,
    TensorArgument,
)
from torch.fx import _pytree as fx_pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils import _pytree as pytree

Val = Any


def _get_updated_range_constraints(gm):
    def get_shape_env(gm):
        vals = [
            node.meta["val"]
            for node in gm.graph.nodes
            if node.meta.get("val", None) is not None
        ]
        from torch._guards import detect_fake_mode  # type: ignore[21]

        fake_mode = detect_fake_mode(vals)
        if fake_mode is not None:
            return fake_mode.shape_env
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env

    shape_env = get_shape_env(gm)
    if shape_env is None:
        return {}
    range_constraints = {
        k: v
        for k, v in shape_env.var_to_range.items()
        if k not in shape_env.replacements
    }
    # Only when we have an unbacked symint, and it's used as constructor inputs,
    # runtime_var_to_range will make a difference compated to var_to_range.
    # e.g. [2, oo) -> [0, oo)
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements:
            range_constraints[k] = v
    return range_constraints


def _get_updated_graph_signature(
    old_signature: ExportGraphSignature,
    new_gm: torch.fx.GraphModule,
) -> ExportGraphSignature:
    """
    Update the graph signature's user_input/user_outputs.
    """
    new_input_specs = []
    i = 0
    for node in new_gm.graph.nodes:
        if node.op != "placeholder":
            continue

        assert i < len(
            old_signature.input_specs
        ), "Number of inputs changed after transformation"
        old_input_spec = old_signature.input_specs[i]
        arg = (
            old_input_spec.arg
            if isinstance(old_input_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_input_spec.arg)(node.name)
        )
        new_input_specs.append(
            InputSpec(
                old_input_spec.kind,
                arg,
                old_input_spec.target,
                persistent=old_input_spec.persistent,
            )
        )
        i += 1

    output_node = list(new_gm.graph.nodes)[-1]
    assert output_node.op == "output"

    new_output_specs = []
    for i, node in enumerate(output_node.args[0]):
        assert i < len(
            old_signature.output_specs
        ), "Number of outputs changed after transformation"
        old_output_spec = old_signature.output_specs[i]
        arg = (
            old_output_spec.arg
            if isinstance(old_output_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_output_spec.arg)(node.name)
        )
        new_output_specs.append(
            OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
        )

    new_signature = ExportGraphSignature(
        input_specs=new_input_specs, output_specs=new_output_specs
    )
    return new_signature


def _transform(self, *passes: PassType) -> "ExportedProgram":
    pm = PassManager(list(passes))
    res = pm(self.graph_module)
    transformed_gm = res.graph_module if res is not None else self.graph_module
    assert transformed_gm is not None

    if transformed_gm is self.graph_module and not res.modified:
        return self

    transformed_ep = ExportedProgram(
        root=transformed_gm,
        graph=transformed_gm.graph,
        graph_signature=_get_updated_graph_signature(
            self.graph_signature, transformed_gm
        ),
        state_dict=self.state_dict,
        range_constraints=_get_updated_range_constraints(transformed_gm),
        module_call_graph=copy.deepcopy(self._module_call_graph),
        example_inputs=self.example_inputs,
        verifier=self.verifier,
        constants=self.constants,
    )
    transformed_ep.graph_module.meta.update(self.graph_module.meta)
    transformed_ep.graph_module.meta.update(res.graph_module.meta)
    return transformed_ep


def _copy_module(new_prog, new_gm):
    new_prog.meta.update(new_gm.meta)
    new_prog.graph = new_gm.graph
    submodules = [name for name, _ in new_prog.named_children()]
    for name in submodules:
        delattr(new_prog, name)
    for name, mod in new_gm.named_children():
        setattr(new_prog, name, mod)
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            t = getattr(new_gm, node.target, None)
            if isinstance(t, torch.Tensor):
                setattr(new_prog, node.target, t)


def lift_constant_tensor_pass(ep):
    """
    Takes an ExportedProgram and returns the ExportedProgram modified in-place,
    with the constant tensors as buffers.
    """
    if len([node for node in ep.graph.nodes if node.op == "placeholder"]) == 0:
        return ep

    graph_signature = ep.graph_signature
    buffers = graph_signature.buffers

    fake_mode = list(ep.graph.nodes)[0].meta["val"].fake_mode
    first_user_input = None
    lifted_constants = []
    for node in ep.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break

    for node in ep.graph.nodes:
        if node.op == "get_attr":
            constant_tensor = getattr(ep.graph_module, node.target)
            if not isinstance(constant_tensor, torch.Tensor):
                continue

            constant_tensor_fqn = f"_lifted_tensor_constant{len(buffers)}"

            with ep.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = ep.graph.placeholder(constant_tensor_fqn)
                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v
                if fake_mode is not None:
                    const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                        constant_tensor, static_shapes=True
                    )
                else:
                    const_placeholder_node.meta["val"] = constant_tensor
                const_placeholder_node.meta["val"].constant = constant_tensor
                node.replace_all_uses_with(const_placeholder_node)
                ep.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                lifted_constants.append(
                    InputSpec(
                        kind=InputKind.BUFFER,
                        arg=TensorArgument(name=const_placeholder_node.name),
                        target=constant_tensor_fqn,
                        persistent=True,
                    )
                )
                buffers.append(constant_tensor_fqn)
                ep.state_dict[constant_tensor_fqn] = constant_tensor

    new_input_specs = []
    for s in graph_signature.input_specs:
        if s.kind == InputKind.USER_INPUT and len(lifted_constants) > 0:
            new_input_specs.extend(lifted_constants)
            lifted_constants.clear()
        new_input_specs.append(s)
    ep.graph_signature.input_specs = new_input_specs
    ep.graph_module.recompile()
    return ep


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
        module_call_graph,
        example_inputs,
        verifier,
    ):
        super().__init__(
            root=root,
            graph=graph,
            graph_signature=graph_signature,
            state_dict=state_dict,
            range_constraints=range_constraints,
            module_call_graph=module_call_graph,
            example_inputs=example_inputs,
            verifier=verifier,
        )

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
        self.exported_program = _transform(self.exported_program, *passes)
        return self

    def __call__(self, *args: Any) -> Any:
        return self.exported_program.module()(*args)

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
        new_gm = self.exported_program.graph_module
        for p in edge_to_executorch_passes(config):
            new_gm_res = p(new_gm)
            assert new_gm_res is not None
            new_gm = new_gm_res.graph_module

        # This is tech debt on tech debt. memory planning pass inherits from some pass infra for GMs.
        # This isnt enough info now so i cant use call I have to use some new function 'run'.
        # Existing user passes dont use run so Im just cheating here because they dont need to work on mutable buffers yet.
        # After exir.capture is gone I will clean up the memory planning infra to be consistent.
        # Frankly all of exir has big code quality issues because of the migrations that need to be addressed.
        new_gm_res = config.memory_planning_pass(new_gm)  # pyre-ignore[19]
        assert new_gm_res is not None
        new_gm = new_gm_res.graph_module
        new_prog = ExirExportedProgram(
            copy.deepcopy(self.exported_program), self.after_to_edge_passes
        )
        _copy_module(new_prog.exported_program.graph_module, new_gm)
        executorch_prog = ExecutorchProgram(
            new_prog,
            emit_stacktrace=config.emit_stacktrace,
            extract_delegate_segments=config.extract_delegate_segments,
            extract_constant_segment=config.extract_constant_segment,
            segment_alignment=config.segment_alignment,
            constant_tensor_alignment=config.constant_tensor_alignment,
            delegate_alignment=config.delegate_alignment,
        )
        executorch_prog.graph_module.meta.update(new_gm.meta)
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
        extract_delegate_segments: bool,
        extract_constant_segment: bool,
        segment_alignment: int,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
    ) -> None:
        if not exir_exported_program.after_to_edge_passes:
            raise RuntimeError(
                "Need to call prog.to_edge prior to constructing ExecutorchProgram."
            )
        self.exported_program = exir_exported_program.exported_program
        self._pte_data: Optional[Cord] = None
        self._buffer: Optional[bytes] = None
        self._emitter_output: Optional[EmitterOutput] = None
        self._emit_stacktrace: bool = emit_stacktrace
        self._extract_delegate_segments: bool = extract_delegate_segments
        self._extract_constant_segment: bool = extract_constant_segment
        self._segment_alignment: int = segment_alignment
        self._constant_tensor_alignment: Optional[int] = constant_tensor_alignment
        self._delegate_alignment: Optional[int] = delegate_alignment

    def _get_pte_data(self) -> Cord:
        if self._pte_data is None:
            self._pte_data = _serialize_pte_binary(
                program=self.program,
                extract_delegate_segments=self._extract_delegate_segments,
                extract_constant_segment=self._extract_constant_segment,
                segment_alignment=self._segment_alignment,
                constant_tensor_alignment=self._constant_tensor_alignment,
                delegate_alignment=self._delegate_alignment,
            )
        return self._pte_data

    @property
    def buffer(self) -> bytes:
        """Returns the serialized ExecuTorch binary as a byte string.

        Note that the call to `buffer` may allocate a very large amount of
        contiguous memory, depending on the model size. If writing to a file,
        use `write_to_file` which won't incur additional copies.
        """
        # TODO(T181494963): update pybinding to remove buffer cache, which can consume large
        # amounts of memory longer than necessary.
        if self._buffer is None:
            self._buffer = bytes(self._get_pte_data())
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

    def write_to_file(self, open_file: io.BufferedIOBase) -> None:
        """
        Writes the serialized ExecuTorch binary to the file at `open_file`. Prefer to use this over
        `buffer`, as it writes to file without copying into a contiguous block of memory first,
        reducing the peak memory usage.
        """
        self._get_pte_data().write_to_file(open_file)


def _get_aten_to_edge_passes(config: EdgeCompileConfig):
    # TODO: the last two passes for aten_to_edge need to be eliminated_dead_code -> debug_handle_generator. After enable
    # use_edge_op it can be moved to aten_to_edge_passes before eliminated_dead_code pass. Also ExportPass doesn't play
    # well with node.meta, meaning after some passes permuting operators, we may lose some information in node.meta.
    # It might be regenerated in SpecPropPass so it may not be visiable. However debug handle will be lost.

    pre_op_replace_passes = base_pre_op_replace_passes + (
        [] if config._skip_type_promotion else [RemoveMixedTypeOperators()]
    )

    post_op_replace_passes = (
        [] if config._skip_dim_order else [MemoryFormatOpsPass()]
    ) + base_post_op_replace_passes

    return pre_op_replace_passes, post_op_replace_passes


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

    dialect = ep.exported_program.dialect
    if dialect == "ATEN":
        ep = ExirExportedProgram(
            ExportedProgram(
                root=ep.exported_program.graph_module,
                graph=ep.exported_program.graph_module.graph,
                graph_signature=ep.exported_program.graph_signature,
                state_dict=ep.exported_program.state_dict,
                range_constraints=ep.exported_program.range_constraints,
                module_call_graph=ep.exported_program.module_call_graph,
                example_inputs=ep.exported_program.example_inputs,
                verifier=get_aten_verifier(enable=config._check_ir_validity),
                constants=ep.exported_program.constants,
            ),
            False,
        )
    pre_op_replace_passes, post_op_replace_passes = _get_aten_to_edge_passes(config)

    new_ep = copy.deepcopy(ep).transform(*pre_op_replace_passes)
    if dialect == "ATEN":
        new_ep.exported_program = lift_constant_tensor_pass(new_ep.exported_program)

    new_gm = new_ep.exported_program.graph_module
    if config._use_edge_ops:
        new_gm_res = OpReplacePass()(new_gm)
        assert new_gm_res is not None
        new_gm = new_gm_res.graph_module

    for p in post_op_replace_passes:
        new_gm_res = p(new_gm)
        assert new_gm_res is not None
        new_gm = new_gm_res.graph_module

    new_ep.exported_program = ExportedProgram(
        root=new_gm,
        graph=new_gm.graph,
        graph_signature=_get_updated_graph_signature(
            new_ep.exported_program.graph_signature, new_gm
        ),
        state_dict=new_ep.exported_program.state_dict,
        range_constraints=new_ep.exported_program.range_constraints,
        module_call_graph=new_ep.exported_program.module_call_graph,
        example_inputs=new_ep.exported_program.example_inputs,
        verifier=EXIREdgeDialectVerifier(
            check_edge_ops=config._use_edge_ops,
            enable=config._check_ir_validity,
            class_only=True,
        ),
        constants=new_ep.exported_program.constants,
    )
    new_ep.after_to_edge_passes = True
    return new_ep


def pre_memory_planning_passes(config: ExecutorchBackendConfig) -> List[PassType]:
    if config.remove_view_copy:
        # pyre-ignore
        return [
            NormalizeViewCopyBasePass(),
            dead_code_elimination_pass,
            ReplaceViewCopyWithViewPass(),
            config.sym_shape_eval_pass,
            config.to_out_var_pass,
        ]
    else:
        # pyre-ignore
        return [
            config.sym_shape_eval_pass,
            config.to_out_var_pass,
        ]


def edge_to_executorch_passes(config: ExecutorchBackendConfig) -> List[PassType]:
    passes: List[PassType] = [
        *config.passes,
        SpecPropPass(),
        # ExecuTorch backend ops are unable to handle unbacked symints. So after
        # this pass, passes cannot be Interpreter-based, because it will fail if
        # there exists an unbacked symint operation.
        EdgeToBackendOpsPass(),
        RemoveGraphAssertsPass(),
    ] + pre_memory_planning_passes(config)

    return passes


def to_edge(
    programs: Union[ExportedProgram, Dict[str, ExportedProgram]],
    constant_methods: Optional[Dict[str, Any]] = None,
    compile_config: Optional[EdgeCompileConfig] = None,
) -> "EdgeProgramManager":
    """
    :func:`to_edge` constructs an EdgeProgramManager from a set of exported programs in
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
        program = program.run_decompositions(_default_decomposition_table())

        if config._check_ir_validity:
            try:
                EXIRATenDialectVerifier()(program.graph_module)
            except ExportError as e:
                logging.info(f"Input program {name} is not in ATen dialect.")
                raise e

        pre_op_replace_passes, post_op_replace_passes = _get_aten_to_edge_passes(config)

        passes = []
        passes.append(
            ReplaceViewOpsWithViewCopyOpsPass()
        )  # TODO move inside aten_to_edge passes after all users are migrated off v1 capture
        passes.extend(pre_op_replace_passes)
        if config._use_edge_ops:
            passes.append(OpReplacePass())

        gm = program.graph_module
        for p in passes:
            gm_res = p(gm)
            assert gm_res is not None
            gm = gm_res.graph_module

        edge_program = ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=_get_updated_graph_signature(program.graph_signature, gm),
            state_dict=program.state_dict,
            range_constraints=program.range_constraints,
            module_call_graph=program.module_call_graph,
            example_inputs=program.example_inputs,
            verifier=EXIREdgeDialectVerifier(
                check_edge_ops=config._use_edge_ops,
                enable=config._check_ir_validity,
                class_only=True,
            ),
            constants=program.constants,
        )
        # Lift the tensor constants created in ScalarToTensorPass
        edge_program = lift_constant_tensor_pass(edge_program)
        edge_program = _transform(edge_program, *post_op_replace_passes)
        edge_programs[name] = edge_program
    return EdgeProgramManager(edge_programs, constant_methods, config)


class EdgeProgramManager:
    """
    Package of one or more `ExportedPrograms` in Edge dialect. Designed to simplify
    lowering to ExecuTorch. See: https://pytorch.org/executorch/stable/ir-exir.html

    Allows easy applications of transforms across a collection of exported programs
    including the delegation of subgraphs.

    Manages the second link in the lowering chain of ATen -> Edge -> ExecuTorch.
    """

    def __init__(
        self,
        edge_programs: Dict[str, ExportedProgram],
        constant_methods: Optional[Dict[str, Any]] = None,
        compile_config: Optional[EdgeCompileConfig] = None,
    ):
        """
        Should not be called directly by users. User should use :func:'to_edge' instead.

        Constructs an EdgeProgramManager from an existing set of exported programs in edge dialect.
        """
        config = compile_config or EdgeCompileConfig()
        for name, program in edge_programs.items():
            try:
                EXIREdgeDialectVerifier(
                    check_edge_ops=config._use_edge_ops,
                    enable=config._check_ir_validity,
                )(program.graph_module)
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
        check_ir_validity: bool = True,
        # We should also probably add check_edge_ops here as well
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
                    new_programs[name] = _transform(program, *passes[name])
                    EXIREdgeDialectVerifier(enable=check_ir_validity)(
                        new_programs[name].graph_module
                    )
                else:
                    new_programs[name] = copy.deepcopy(program)

        else:  # apply passes to every method
            for name, program in self._edge_programs.items():
                new_programs[name] = _transform(program, *passes)
                EXIREdgeDialectVerifier(enable=check_ir_validity)(
                    new_programs[name].graph_module
                )
        config = EdgeCompileConfig(_check_ir_validity=check_ir_validity)
        return EdgeProgramManager(
            new_programs, copy.deepcopy(self._config_methods), config
        )

    def to_backend(
        self, partitioner: Union[Partitioner, Dict[str, Partitioner]]
    ) -> "EdgeProgramManager":
        """
        Returns a semantically-equivalent program to the one given as input,
        but with portions of each program in the EdgeProgramManager targeted
        for delegation as determined by the partitioner.

        Args:
            partitioner: The partitioner can either be a Partitioner subclass instance, or a
                dictionary mapping method names to Partitioner subclass instance. If it is a
                Partitioner subclass, all programs in the given EdgeProgramManager
                will be lowered using the given partitioner. If it is a
                dictionary, only method names specified in the dictionary will be
                lowered with the given partitioner.

                The Partitioner subclass instance is in charge with tagging portions of the
                input program for delegation. A valid partitioner must return PartitionerResult including valid
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

        config = EdgeCompileConfig(_check_ir_validity=False)
        return EdgeProgramManager(
            new_edge_programs, copy.deepcopy(self._config_methods), config
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
            program = unsafe_remove_auto_functionalized_pass(program)
            gm, new_signature = insert_write_back_for_buffers_pass(program)
            new_gm = program.graph_module
            for p in edge_to_executorch_passes(config):
                new_gm_res = p(new_gm)
                assert new_gm_res is not None
                new_gm = new_gm_res.graph_module
                if isinstance(p, SpecPropPass):
                    # Note that this is a hacky way to get around the fact that
                    # placeholder nodes corresponding to the parameters of the graph module
                    # shall not participate in memory planning. It increases runtime memory
                    # footprint.
                    # Proper way would be to have ExportPass work with ExportedProgram
                    # instead of GraphModule. This is because ExportPass should work
                    # on top of the export artifact of torch.export whichi s ExportedProgram.
                    # Working with GraphModule does not provide all the information contained
                    # in the ExportedProgram
                    # TODO(who?)
                    p.update_placeholder_tensor_specs(program, new_gm)

            # TODO(jakeszwe): Follow up with compiler on if the deepcopy is necessary and if so how to make it work
            if hasattr(config.memory_planning_pass, "run"):
                new_gm_res = config.memory_planning_pass.run(  # pyre-ignore[16]
                    new_gm, new_signature
                )
            else:
                new_gm_res = config.memory_planning_pass(new_gm)  # pyre-ignore[19]
            assert new_gm_res is not None
            new_gm = new_gm_res.graph_module

            _copy_module(program.graph_module, new_gm)
            execution_programs[name] = program

        return ExecutorchProgramManager(
            execution_programs, self._config_methods, config
        )


class ExecutorchProgramManager:
    """
    Package of one or more `ExportedPrograms` in Execution dialect. Designed to simplify
    lowering to ExecuTorch. See: https://pytorch.org/executorch/stable/ir-exir.html

    When the ExecutorchProgramManager is constructed the ExportedPrograms in execution dialect
    are used to form the executorch binary (in a process called emission) and then serialized
    to a buffer.

    Manages the final link in the lowering chain of ATen -> Edge -> ExecuTorch.
    """

    def __init__(
        self,
        execution_programs: Dict[str, ExportedProgram],
        config_methods: Optional[Dict[str, Any]] = None,
        backend_config: Optional[ExecutorchBackendConfig] = None,
    ):
        """
        End users should not call this constructor directly. Instead, they should use
        :func:'to_executorch' to construct an ExecutorchProgramManager.

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

        # Serialize emitter output, ready to be written to a file.
        self._pte_data: Cord = _serialize_pte_binary(
            program=self._emitter_output.program,
            extract_delegate_segments=backend_config.extract_delegate_segments,
            extract_constant_segment=backend_config.extract_constant_segment,
            segment_alignment=backend_config.segment_alignment,
            constant_tensor_alignment=backend_config.constant_tensor_alignment,
            delegate_alignment=backend_config.delegate_alignment,
        )
        self._buffer: Optional[bytes] = None

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
        """Returns the serialized ExecuTorch binary as a byte string.

        Note that the call to `buffer` may allocate a very large amount of
        contiguous memory, depending on the model size. If writing to a file,
        use `write_to_file` which won't incur additional copies.
        """
        # TODO(T181494963): update pybinding to remove buffer cache, which can consume large
        # amounts of memory longer than necessary.
        if self._buffer is None:
            self._buffer = bytes(self._pte_data)
        return self._buffer

    def write_to_file(self, open_file: io.BufferedIOBase) -> None:
        """
        Writes the serialized ExecuTorch binary to the file at `open_file`. Prefer to use this over
        `buffer`, as it writes to file without copying into a contiguous block of memory first,
        reducing the peak memory usage.
        """
        self._pte_data.write_to_file(open_file)
