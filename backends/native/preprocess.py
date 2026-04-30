# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NativeBackend — AOT BackendDetails for the v2 portable runtime.

The class name `NativeBackend` is the backend_id used at runtime
to find the matching BackendInterface. C++ side registers it via
`register_backend({"NativeBackend", ...})` in
native/NativeBackend.cpp.

The preprocess() pipeline:
1. SpecPropPass to populate tensor specs.
2. NEW: reinplace_pass with the backend's BACKEND_INPLACE_OPS registry
   (~85 edge-functional ops mapped to aten in-place targets). Rewrites
   `op(self, ...)` -> `op_(self, ...)` when ET's safety check passes.
   Re-runs SpecPropPass so new in-place nodes get spec metadata.
3. (If buffer mutations exist) insert_write_back_for_buffers_pass to
   add an explicit aten::copy_(buf, mut_src) at end of subgraph. When
   step 2 already rewrote the buffer mutation in-place,
   `_inplace_lineage` detects this and skips inserting the writeback.
4. Spec-sharing fallback for any remaining writebacks: alias the
   mutation source's TensorSpec to the buffer's. Used only when step 2
   didn't catch the buffer mutation (e.g., custom op outside the
   registry).
5. ExternalConstantsPass to tag constants for NDM storage.
6. Memory planning (greedy, allow_overlapping_allocations). The
   upstream `_alias_inplace_result_specs` aliases in-place op result
   specs to mutated input specs so emit's `_emit_spec` dedup gives
   them one value_id.
7. Emit and serialize.
"""

from functools import partial
from typing import Any, Dict, final, List

import torch

from executorch.exir._serialize._program import PTEFile, serialize_pte_binary

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

from torch._export.verifier import Verifier


# ---------------------------------------------------------------------------
# Backend-owned in-place op registry (consumed by ET's reinplace_pass).
#
# Maps edge-dialect functional ops -> aten in-place targets. The pass
# runs on the edge-dialect graph; targets use the aten in-place form
# because the edge dialect doesn't register in-place variants for most
# ops. The lowered IR ends up carrying `aten::<name>_` instructions
# which the executor accepts.
#
# Built once at module load by intersecting `ops.edge.aten.<name>` with
# `torch.ops.aten.<name>_`. This table lives in the backend (not
# upstream) because it's a per-runtime kernel-availability statement.
# ---------------------------------------------------------------------------


def _build_backend_inplace_ops() -> Dict[Any, Any]:
    """Construct the v2 backend's in-place rewrite registry by listing
    the edge-functional ops we support and letting ET's reinplace_pass
    infrastructure schema-match each to its aten in-place counterpart.

    The list itself is just "what edge-dialect functional ops do we
    support an in-place form for?" — overload-name asymmetries (e.g.,
    `pow.Tensor_Scalar` ↔ `pow_.Scalar`) are handled automatically by
    `find_inplace_overload` via schema matching.
    """
    from executorch.exir.dialects._ops import ops as _edge_ops
    from executorch.exir.passes.reinplace import (
        DEFAULT_INPLACEABLE_OPS,
        build_inplace_registry_for,
    )

    edge = _edge_ops.edge.aten

    # (functional_name, [overload_name, ...]) — list every edge functional
    # op that should be considered for in-place rewriting. The schema
    # matcher in `find_inplace_overload` figures out the correct in-place
    # overload (which may have a different overload name than the
    # functional one).
    pairs = [
        # ------- pointwise unary -------
        ("relu", ["default"]), ("relu6", ["default"]),
        ("sigmoid", ["default"]), ("tanh", ["default"]),
        ("exp", ["default"]), ("expm1", ["default"]),
        ("log", ["default"]), ("log1p", ["default"]),
        ("log2", ["default"]), ("log10", ["default"]),
        ("neg", ["default"]), ("abs", ["default"]),
        ("sqrt", ["default"]), ("rsqrt", ["default"]),
        ("reciprocal", ["default"]), ("square", ["default"]),
        ("cos", ["default"]), ("sin", ["default"]),
        ("tan", ["default"]), ("cosh", ["default"]),
        ("sinh", ["default"]), ("asin", ["default"]),
        ("acos", ["default"]), ("atan", ["default"]),
        ("asinh", ["default"]), ("acosh", ["default"]),
        ("atanh", ["default"]),
        ("erf", ["default"]), ("erfc", ["default"]),
        ("sign", ["default"]),
        ("ceil", ["default"]), ("floor", ["default"]),
        ("round", ["default"]), ("trunc", ["default"]),
        ("frac", ["default"]),
        ("silu", ["default"]), ("gelu", ["default"]),
        ("elu", ["default"]), ("leaky_relu", ["default"]),
        ("hardtanh", ["default"]), ("hardsigmoid", ["default"]),
        ("hardswish", ["default"]),
        ("logical_not", ["default"]), ("bitwise_not", ["default"]),
        # ------- binary -------
        ("add", ["Tensor", "Scalar"]), ("sub", ["Tensor", "Scalar"]),
        ("mul", ["Tensor", "Scalar"]), ("div", ["Tensor", "Scalar"]),
        # `pow.Tensor_Scalar` (Tensor self, Scalar exp) is what `square_`
        # decomposes to. Schema-matching finds its in-place counterpart
        # `pow_.Scalar` (overload name differs from functional).
        ("pow", ["Scalar", "Tensor_Scalar"]),
        ("remainder", ["Tensor", "Scalar"]),
        ("fmod", ["Tensor", "Scalar"]),
        ("atan2", ["default"]),
        ("logical_and", ["default"]), ("logical_or", ["default"]),
        ("logical_xor", ["default"]),
        ("bitwise_and", ["Tensor", "Scalar"]),
        ("bitwise_or", ["Tensor", "Scalar"]),
        ("bitwise_xor", ["Tensor", "Scalar"]),
        # ------- scatter / index -------
        ("index_copy", ["default"]),
        ("index_fill", ["int_Scalar"]),
        ("index_add", ["default"]),
        ("scatter", ["src", "value"]),
        ("scatter_add", ["default"]),
        ("masked_fill", ["Scalar", "Tensor"]),
        ("masked_scatter", ["default"]),
        # ------- misc -------
        ("fill", ["Scalar", "Tensor"]),
        ("clamp", ["default"]), ("clamp_min", ["default"]),
        ("clamp_max", ["default"]),
        ("addcmul", ["default"]), ("addcdiv", ["default"]),
        ("lerp", ["Scalar", "Tensor"]),
    ]

    # Resolve our (name, overload) pairs to actual edge OpOverloads.
    functional_ops = []
    for name, overloads in pairs:
        edge_pkg = getattr(edge, name, None)
        if edge_pkg is None:
            continue
        for ovld in overloads:
            op = getattr(edge_pkg, ovld, None)
            if op is not None:
                functional_ops.append(op)

    # Schema-matched discovery handles all overload-name asymmetries.
    registry: Dict[Any, Any] = dict(DEFAULT_INPLACEABLE_OPS)  # start with index_put
    registry.update(build_inplace_registry_for(functional_ops))
    return registry


# Cached at module load. Backends can extend this dict directly if they
# add new in-place kernels to their runtime.
BACKEND_INPLACE_OPS: Dict[Any, Any] = _build_backend_inplace_ops()


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
    options: Dict[str, Any] = {
        "skip_memory_planning": False,
        # Default ON: the v2 portable runtime's CPU provider has
        # dispatches for in-place ops (`aten::add_` etc.) that route to
        # the existing `.out` variant kernels. The router falls back to
        # CPU when no other provider claims the op.
        "enable_reinplace": True,
    }
    for spec in compile_specs:
        if spec.key in options and isinstance(options[spec.key], bool):
            options[spec.key] = bool.from_bytes(spec.value, byteorder="little")
    return options


@final
class NativeBackend(BackendDetails):
    """
    BackendDetails for the v2 portable backend.

    Class name `NativeBackend` matches the runtime backend_id
    registered in native/NativeBackend.cpp.
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
        skip_memory_planning = compile_options["skip_memory_planning"]
        enable_reinplace = compile_options["enable_reinplace"]

        # Step 1: SpecPropPass to propagate tensor specs.
        program = _apply_passes(program, [SpecPropPass()])

        # Step 2: NEW — reinplace_pass with the backend's in-place op
        # registry. Rewrites functional ops to their `*_` form when ET's
        # safety check passes (sole consumer; mutable inputs OK; no
        # later use). For buffer mutations (KV-cache pattern), this
        # turns `index_put(buf, ...)` into `index_put_(buf, ...)`,
        # which then makes `insert_write_back_for_buffers_pass` (step 3)
        # skip inserting a redundant `copy_` writeback via its
        # `_inplace_lineage` check.
        if enable_reinplace:
            from executorch.exir.passes.reinplace import (
                reinplace_pass as _et_reinplace_pass,
            )

            program = _et_reinplace_pass(
                program, ops_to_inplace=BACKEND_INPLACE_OPS
            )
            # ET's reinplace_pass copies meta["val"] to the new in-place
            # node but not meta["spec"]. Re-run SpecPropPass so the new
            # nodes have specs that downstream passes can read.
            program = _apply_passes(program, [SpecPropPass()])

        # Step 3: Insert writeback copy_ ops for any mutable buffers
        # that weren't already mutated in place by step 2. The pass
        # uses `_inplace_lineage` to detect in-place chains and skip
        # writeback insertion for them.
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
                # Alias: src now shares buf's spec object.
                src_node.meta["spec"] = buf_spec

        # Step 4: External constants pass — tag weights for NDM storage.
        from executorch.exir.passes.external_constants_pass import (
            external_constants_pass,
        )

        external_constants_pass(program.graph_module)

        # Step 5: Memory planning (greedy, allows overlapping allocs).
        if not skip_memory_planning:
            greedy_memory_planning = partial(greedy, allow_overlapping_allocations=True)
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

        # Step 6: Emit the program.
        delegate_mapping_builder = DelegateMappingBuilder(generated_identifiers=True)
        emitter_output = emit_program(program)

        # Step 7: Build named data store from external constants.
        from executorch.exir._serialize._named_data_store import NamedDataStore

        named_data_store = NamedDataStore()
        if emitter_output.external_constant_buffer:
            for tag, fqn_to_idx in emitter_output.external_constant_map.items():
                for fqn, idx in fqn_to_idx.items():
                    data = emitter_output.external_constant_buffer[idx]
                    named_data_store.add_named_data(fqn, data)

        # Step 8: Serialize to bytes.
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
