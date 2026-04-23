# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PortableBackend_v2 — AOT BackendDetails for the v2 portable runtime.

The class name `PortableBackend_v2` is the backend_id used at runtime
to find the matching BackendInterface. C++ side registers it via
`register_backend({"PortableBackend_v2", ...})` in
runtime_v2/PortableBackend_v2.cpp.

The preprocess() pipeline:
1. SpecPropPass to populate tensor specs.
2. (If buffer mutations exist) insert_write_back_for_buffers_pass to
   add an explicit aten::copy_(buf, mut_src) at end of subgraph.
3. Spec-sharing for buffer mutations: make the mutation source's
   TensorSpec be the SAME object as the buffer placeholder's TensorSpec.
   The greedy memory planner walks specs by identity, so this yields a
   single allocation slot for both — true in-place buffer mutation.
   At runtime, the trailing aten::copy_ becomes a self-copy
   (dispatcher's pointer-equality short-circuit makes it a no-op).
4. ExternalConstantsPass to tag constants for NDM storage.
5. Memory planning (greedy, allow_overlapping_allocations).
6. Emit and serialize.
"""

from functools import partial
from typing import Any, Dict, final, List

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.emit import emit_program
from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
from executorch.exir.passes import MemoryPlanningPass, SpecPropPass
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.program._program import _transform
from executorch.exir._serialize._program import serialize_pte_binary, PTEFile

from torch._export.verifier import Verifier


class _AnyOp(Verifier):
    """Permissive verifier that allows any op (skip functional check)."""

    dialect = "TRAINING"

    def allowed_op_types(self):
        from typing import Callable

        return (Callable,)


def _apply_passes(program: ExportedProgram, passes) -> ExportedProgram:
    """Apply a sequence of passes to an ExportedProgram."""
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


def _parse_compile_spec(compile_specs: List[CompileSpec]) -> Dict[str, Any]:
    """Parse compile specs into options dict."""
    options = {}
    for spec in compile_specs:
        if spec.key == "skip_memory_planning":
            options[spec.key] = bool.from_bytes(spec.value, byteorder="little")
    return options


@final
class PortableBackend_v2(BackendDetails):
    """
    BackendDetails for the v2 portable backend.

    Class name `PortableBackend_v2` matches the runtime backend_id
    registered in runtime_v2/PortableBackend_v2.cpp.
    """

    @classmethod
    def preprocess(
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        """
        Preprocess the partitioned subgraph for v2 portable backend execution.
        """
        compile_options = _parse_compile_spec(module_compile_spec)
        skip_memory_planning = compile_options.get("skip_memory_planning", False)

        # Step 1: SpecPropPass to propagate tensor specs.
        program = _apply_passes(program, [SpecPropPass()])

        # Step 1b: Insert writeback copy_ ops for any mutable buffers that
        # got pulled into our delegate by the partitioner's
        # tag_mutated_buffer call. Then alias the mutation output's spec
        # with the buffer placeholder's spec — so the AOT memory planner
        # treats them as a single tensor and allocates ONE slot, achieving
        # true in-place buffer mutation. The trailing aten::copy_ becomes
        # a self-copy at runtime (handled by dispatcher's pointer check).
        from torch.export.graph_signature import InputKind, OutputKind

        has_buffer_mutation = any(
            ospec.kind == OutputKind.BUFFER_MUTATION
            for ospec in program.graph_signature.output_specs
        )
        if has_buffer_mutation:
            gm, new_sig = insert_write_back_for_buffers_pass(program)
            program._graph_module = gm
            program._graph_signature = new_sig
            # Re-propagate specs onto the newly inserted copy_ nodes.
            program = _apply_passes(program, [SpecPropPass()])

            # Spec-sharing trick: for each (buffer_placeholder, mutation
            # source) pair, make the mutation source's TensorSpec be the
            # SAME object as the buffer's TensorSpec. The greedy memory
            # planner walks specs by identity, so a shared spec yields a
            # single allocation. No wasted slot, no runtime override
            # needed (the .pte naturally reports both at the same offset).
            import torch

            sig = program.graph_signature
            nodes_by_name = {
                n.name: n for n in program.graph_module.graph.nodes
            }
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
                # Alias: src now shares buf's spec object.
                src_node.meta["spec"] = buf_spec

        # Step 2: External constants pass — tag weights for NDM storage.
        from executorch.exir.passes.external_constants_pass import (
            external_constants_pass,
        )

        external_constants_pass(program.graph_module)

        # Step 3: Memory planning (greedy, allows overlapping allocs).
        if not skip_memory_planning:
            greedy_memory_planning = partial(
                greedy, allow_overlapping_allocations=True
            )
            mem_planning_suite = MemoryPlanningAlgorithmSuite(
                algo_list=[greedy_memory_planning]
            )

            # Workaround for memory planning without ToOutVarPass
            program.graph_module.encounter_to_out_var_failure = True

            program = _apply_passes(
                program,
                [
                    ConstraintBasedSymShapeEvalPass(),
                    MemoryPlanningPass(memory_planning_algo=mem_planning_suite),
                ],
            )

        # Step 4: Emit the program.
        delegate_mapping_builder = DelegateMappingBuilder(generated_identifiers=True)
        emitter_output = emit_program(program)

        # Step 5: Build named data store from external constants.
        from executorch.exir._serialize._named_data_store import NamedDataStore

        named_data_store = NamedDataStore()
        if emitter_output.external_constant_buffer:
            for tag, fqn_to_idx in emitter_output.external_constant_map.items():
                for fqn, idx in fqn_to_idx.items():
                    data = emitter_output.external_constant_buffer[idx]
                    named_data_store.add_named_data(fqn, data)

        # Step 6: Serialize to bytes.
        pte_file = PTEFile(
            program=emitter_output.program,
            mutable_data=emitter_output.mutable_data,
        )
        serialized_bytes = bytes(serialize_pte_binary(pte_file))

        return PreprocessResult(
            processed_bytes=serialized_bytes,
            debug_handle_map=emitter_output.debug_handle_map,
            data_store_output=named_data_store.get_named_data_store_output(),
        )
