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

import copy
import struct
from functools import partial
from typing import Any, Dict, final, List, Optional, Sequence, Tuple, Type

import torch
from executorch.exir._serialize._flatbuffer_program import _program_to_flatbuffer

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.emit import emit_program
from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
from executorch.exir.passes import MemoryPlanningPass, SpecPropPass
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.program._program import _transform

from torch._export.verifier import Verifier

# Type alias for specialization config: (BackendDetails subclass, compile_specs)
Specialization = Tuple[Type[BackendDetails], List[CompileSpec]]


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


def _build_backend_inplace_ops() -> frozenset:
    """Construct the v2 backend's in-place op set by listing the
    edge-functional ops we support. ET's reinplace_pass auto-derives
    the in-place counterpart for each via schema matching.
    """
    from executorch.exir.dialects._ops import ops as _edge_ops

    edge = _edge_ops.edge.aten

    pairs = [
        ("relu", ["default"]),
        ("gelu", ["default"]),
        ("sigmoid", ["default"]),
        ("index_put", ["default"]),
        ("index_copy", ["default"]),
    ]

    functional_ops = set()
    for name, overloads in pairs:
        edge_pkg = getattr(edge, name, None)
        if edge_pkg is None:
            continue
        for ovld in overloads:
            op = getattr(edge_pkg, ovld, None)
            if op is not None:
                functional_ops.add(op)

    return frozenset(functional_ops)


BACKEND_INPLACE_OPS: frozenset = _build_backend_inplace_ops()


# ---------------------------------------------------------------------------
# Fat PTE container format
#
# A fat blob packs multiple backend specializations into one delegate payload.
# The runtime parses the header and picks the best available specialization.
#
# Layout:
#   [4 bytes] magic  "NFAT"
#   [4 bytes] version (1)
#   [4 bytes] num_specializations
#   For each specialization:
#     [32 bytes] backend_id (utf-8, null-padded)
#     [8 bytes]  offset into data section
#     [8 bytes]  length
#   [payload bytes for specialization 0]
#   [payload bytes for specialization 1]
#   ...
# ---------------------------------------------------------------------------

_FAT_MAGIC = b"NFAT"
_FAT_VERSION = 1
_FAT_ENTRY_FMT = "32sQQ"  # backend_id(32) + offset(u64) + size(u64)
_FAT_ENTRY_SIZE = struct.calcsize(_FAT_ENTRY_FMT)


def _pack_fat_blob(
    specializations: List[Tuple[str, bytes]],
) -> bytes:
    """Pack multiple (backend_id, payload) pairs into a fat blob."""
    header = struct.pack("<4sII", _FAT_MAGIC, _FAT_VERSION, len(specializations))

    entries = []
    offset = 0
    for backend_id, payload in specializations:
        bid = backend_id.encode("utf-8")[:32].ljust(32, b"\x00")
        entries.append(struct.pack("<" + _FAT_ENTRY_FMT, bid, offset, len(payload)))
        offset += len(payload)

    return header + b"".join(entries) + b"".join(p for _, p in specializations)


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

    Supports "fat PTE" mode: when specializations are configured,
    preprocess runs each backend's pipeline on the same subgraph and
    packs all results into a single fat blob. The runtime picks the
    best available specialization at load time. All specializations
    share the same PTD (named data store).
    """

    # Set by NativePartitioner before preprocess is called.
    _specializations: Optional[Sequence[Specialization]] = None

    @classmethod
    def preprocess(
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        native_result = cls._preprocess_native(program, module_compile_spec)

        specializations = cls._specializations
        if not specializations:
            return native_result

        # Fat PTE: run each specialization backend and pack results.
        from executorch.exir._serialize._named_data_store import NamedDataStore

        fat_entries: List[Tuple[str, bytes]] = [
            ("NativeBackend", native_result.processed_bytes),
        ]

        merged_data_store = NamedDataStore()
        # Seed with native's named data.
        if native_result.data_store_output:
            for key, entry in native_result.data_store_output.pte_data.items():
                buf = native_result.data_store_output.buffers[entry.buffer_index]
                merged_data_store.add_named_data(key, buf, alignment=entry.alignment)

        for backend_cls, spec_compile_specs in specializations:
            spec_program = copy.deepcopy(program)
            spec_result = backend_cls.preprocess(spec_program, spec_compile_specs)

            fat_entries.append((backend_cls.__name__, spec_result.processed_bytes))

            if spec_result.data_store_output:
                for key, entry in spec_result.data_store_output.pte_data.items():
                    buf = spec_result.data_store_output.buffers[entry.buffer_index]
                    merged_data_store.add_named_data(
                        key, buf, alignment=entry.alignment
                    )

        return PreprocessResult(
            processed_bytes=_pack_fat_blob(fat_entries),
            debug_handle_map=native_result.debug_handle_map,
            data_store_output=merged_data_store.get_named_data_store_output(),
        )

    @classmethod
    def _preprocess_native(
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

            program = _et_reinplace_pass(program, ops_to_inplace=BACKEND_INPLACE_OPS)
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

        # Step 4: External constants pass — tag named parameters/buffers
        # for NDM storage. The stock pass deliberately skips lifted
        # tensor constants ("they are closer to code than data"). We
        # used to override that to force lifted constants to NDM, but
        # the runtime now supports inline constants directly via
        # Engine::ConstRequest.inline_data + Graph::tensor_inline_data,
        # so we keep the stock semantics.
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
        emitter_output = emit_program(program)

        # Step 7: Build named data store from external constants.
        from executorch.exir._serialize._named_data_store import NamedDataStore

        named_data_store = NamedDataStore()
        if emitter_output.external_constant_buffer:
            for _tag, fqn_to_idx in emitter_output.external_constant_map.items():
                for fqn, idx in fqn_to_idx.items():
                    data = emitter_output.external_constant_buffer[idx]
                    named_data_store.add_named_data(fqn, data)

        # Step 8: Serialize the inner program to bytes.
        #
        # We deliberately do NOT use serialize_pte_binary here, because
        # it extracts constants from program.constant_buffer into a
        # separate constant_segment of the OUTER PTE file — which makes
        # the inner program's constant_buffer empty when the runtime
        # parses it. The constant data would end up in segments that
        # the delegate has no view into.
        #
        # _program_to_flatbuffer is the lower-level serializer that
        # keeps program.constant_buffer in place. The runtime's
        # Graph::tensor_inline_data() reads directly from
        # program->constant_buffer()->Get(idx), so inline lifted
        # constants reach Engine::upload_constants via the inline_data
        # path on ConstRequest.
        fb_result = _program_to_flatbuffer(emitter_output.program)
        serialized_bytes = bytes(fb_result.data)

        return PreprocessResult(
            processed_bytes=serialized_bytes,
            debug_handle_map=emitter_output.debug_handle_map,
            data_store_output=named_data_store.get_named_data_store_output(),
        )
