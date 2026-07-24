# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NativeBackend — AOT preprocess for the native portable runtime.

Pipeline: SpecProp → reinplace → writeback → external constants →
memory planning → emit → serialize via _program_to_flatbuffer.
"""

import json
from functools import partial
from typing import final, List, Tuple

import torch

from executorch.backends.native.fat_pte import build_fat_result
from executorch.backends.native.specializations import _SPECIALIZATION_REGISTRY
from executorch.exir._serialize._flatbuffer_program import _program_to_flatbuffer

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)

# A specialization recipe: callable that takes an ExportedProgram and returns
# serialized bytes.  Each recipe owns the full
# to_edge_transform_and_lower → to_executorch → serialize flow.


from executorch.exir.dialects._ops import ops as _edge_ops
from executorch.exir.emit import emit_program
from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
from executorch.exir.passes import MemoryPlanningPass, SpecPropPass
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.program._program import _transform

from torch._export.verifier import Verifier

_edge = _edge_ops.edge.aten
BACKEND_INPLACE_OPS: frozenset = frozenset(
    [
        _edge.relu.default,
        _edge.gelu.default,
        _edge.sigmoid.default,
        _edge.index_put.default,
        _edge.index_copy.default,
    ]
)


class _AnyOp(Verifier):
    # "TRAINING" is the only dialect that allows non-functional (mutating) ops.
    # After reinplace converts e.g. relu → relu_, the verifier must accept them.
    dialect = "TRAINING"

    def allowed_op_types(self):
        from typing import Callable

        return (Callable,)


def _apply_passes(program: ExportedProgram, passes) -> ExportedProgram:
    from executorch.exir.pass_base import ExportPass, PassBase

    for p in passes:
        if isinstance(p, MemoryPlanningPass) and hasattr(p, "run"):
            p.run(program.graph_module)
        elif issubclass(type(p), (ExportPass, PassBase)):
            if hasattr(p, "_exported_program"):
                p._exported_program = program
            program = _transform(program, p, override_verifiers=[_AnyOp])
            if isinstance(p, SpecPropPass):
                p.update_placeholder_tensor_specs(program, program.graph_module)
        else:
            program = p(program)

    return program


@final
class NativeBackend(BackendDetails):
    @classmethod
    def preprocess(
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        names = None
        for spec in module_compile_spec:
            if spec.key == "native_specializations":
                names = json.loads(spec.value.decode("utf-8"))

        if not names:
            return cls._preprocess_native(program)

        for name in names:
            if name not in _SPECIALIZATION_REGISTRY:
                raise ValueError(
                    f"Specialization '{name}' is not registered. "
                    f"Registered: {sorted(_SPECIALIZATION_REGISTRY)}"
                )

        import copy

        native_result = cls._preprocess_native(copy.deepcopy(program))

        results: List[Tuple[str, PreprocessResult]] = [
            ("NativeBackend", native_result),
        ]
        for name in names:
            spec_program = copy.deepcopy(program)
            result = _SPECIALIZATION_REGISTRY[name](spec_program)
            results.append((name, result))

        return build_fat_result(results)

    @classmethod
    def _preprocess_native(
        cls,
        program: ExportedProgram,
    ) -> PreprocessResult:
        program = _apply_passes(program, [SpecPropPass()])

        from executorch.exir.passes.reinplace import (
            reinplace_pass as _et_reinplace_pass,
        )

        program = _et_reinplace_pass(program, ops_to_inplace=BACKEND_INPLACE_OPS)

        # Re-run SpecPropPass: reinplace copies meta["val"] but not meta["spec"].
        program = _apply_passes(program, [SpecPropPass()])

        from torch.export.graph_signature import InputKind, OutputKind

        has_buffer_mutation = any(
            ospec.kind == OutputKind.BUFFER_MUTATION
            for ospec in program.graph_signature.output_specs
        )
        if has_buffer_mutation:
            gm, new_sig = insert_write_back_for_buffers_pass(program)
            program._graph_module = gm
            program._graph_signature = new_sig
            program = _apply_passes(program, [SpecPropPass()])

            # Spec-share buffer placeholders with their mutation sources so the
            # memory planner assigns them the same storage slot.
            sig = program.graph_signature
            nodes_by_name = {n.name: n for n in program.graph_module.graph.nodes}
            buf_target_to_node = {
                ispec.target: nodes_by_name.get(ispec.arg.name)
                for ispec in sig.input_specs
                if ispec.kind == InputKind.BUFFER and ispec.target
            }
            for ospec in sig.output_specs:
                if ospec.kind != OutputKind.BUFFER_MUTATION:
                    continue
                buf_node = buf_target_to_node.get(ospec.target)
                wb_node = nodes_by_name.get(ospec.arg.name)
                if (
                    buf_node is None
                    or wb_node is None
                    or wb_node.op != "call_function"
                    or wb_node.target != torch.ops.aten.copy_.default
                    or len(wb_node.args) < 2
                ):
                    continue
                src_node = wb_node.args[1]
                if not hasattr(src_node, "meta"):
                    continue
                buf_spec = buf_node.meta.get("spec")
                if buf_spec is None or "spec" not in src_node.meta:
                    continue
                src_node.meta["spec"] = buf_spec

        from executorch.exir.passes.external_constants_pass import (
            external_constants_pass,
        )

        external_constants_pass(program.graph_module)

        greedy_memory_planning = partial(greedy, allow_overlapping_allocations=True)
        mem_planning_suite = MemoryPlanningAlgorithmSuite(
            algo_list=[greedy_memory_planning]
        )

        # Tells the emitter to use out-variant kernels for ops that support them.
        program.graph_module.encounter_to_out_var_failure = True

        program = _apply_passes(
            program,
            [
                ConstraintBasedSymShapeEvalPass(),
                MemoryPlanningPass(memory_planning_algo=mem_planning_suite),
            ],
        )

        emitter_output = emit_program(program)

        from executorch.exir._serialize._named_data_store import NamedDataStore

        named_data_store = NamedDataStore()
        if emitter_output.external_constant_buffer:
            for _tag, fqn_to_idx in emitter_output.external_constant_map.items():
                for fqn, idx in fqn_to_idx.items():
                    data = emitter_output.external_constant_buffer[idx]
                    named_data_store.add_named_data(fqn, data)

        # Use _program_to_flatbuffer (not serialize_pte_binary) to keep
        # constants inline — the delegate can't see outer PTE segments.
        fb_result = _program_to_flatbuffer(emitter_output.program)
        serialized_bytes = bytes(fb_result.data)

        return PreprocessResult(
            processed_bytes=serialized_bytes,
            debug_handle_map=emitter_output.debug_handle_map,
            data_store_output=named_data_store.get_named_data_store_output(),
        )
