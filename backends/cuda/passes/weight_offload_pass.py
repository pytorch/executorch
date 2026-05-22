# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EXPERIMENTAL: CUDA weight-offloading pass.

Rewrites every parameter / buffer consumer to read through a
``probe(w, probe_id)`` op, then serializes a v2 payload (schedule
+ floor + pin_fqns + per-FQN dtype / sizes / strides /
storage_offset / nbytes / device) into ``NamedDataStore``. The
runtime half (``backends/cuda/runtime/weight_offload/``) parses
the payload, installs 1-byte GPU dummies in place of AOTI's
constants, and serves probes from a pinned host mirror through a
bounded GPU pool. Single-device only.

Public surface: ``CudaPartitioner(weight_offload=True,
weight_offload_pin_fqns=[...])``.
"""

import struct
import sys

import torch
from torch.library import custom_op, register_fake


_OP_NAMESPACE = "executorch_weight_offload"
_OP_QUALNAME = f"{_OP_NAMESPACE}::probe"


@custom_op(_OP_QUALNAME, mutates_args=())
def probe(w: torch.Tensor, probe_id: int) -> torch.Tensor:
    """Identity in semantics. CUDA backend replaces via c-shim at AOTI compile time.

    Inserted by ``_apply_weight_offload`` before every consumer of every
    parameter (or buffer) placeholder. The CUDA runtime's c-shim
    (``aoti_torch_cuda_probe``) intercepts each call at runtime and
    serves bytes through the bounded GPU pool.

    ``probe_id`` self-identifies each ``(consumer, weight)`` site so the
    runtime needs no schedule cursor — it indexes the schedule directly
    by ``probe_id``. This closes off the entire class of "graph order
    drifted from wrapper order" silent failures that an FX-order cursor
    would have. The pass assigns contiguous ``probe_id`` values 0..N-1
    in graph order; the payload's ``schedule[probe_id] = fqn`` table is
    what the runtime keys on.

    Notes:
      - Eager body returns ``w.clone()`` (not ``w``) so torch's custom-op
        aliasing checker accepts the call — outputs of ``mutates_args=()``
        ops are not allowed to alias inputs. Eager is only exercised by
        tests; the CUDA c-shim is the production path and returns a
        fresh ``SlimTensor`` handle sharing the input's storage (no
        copy).
      - AOTI emits a c-shim call site for every probe in the FX graph
        regardless of CSE — distinct ``probe_id`` arguments make
        otherwise-identical calls syntactically distinct from inductor's
        POV. ``test_weight_offload_probe_dispatch.py`` validates this on
        a multi-consumer model.
      - Weight offloading is mutually exclusive with the CUDA backend's
        ``enable_cuda_graph_for_method`` option: CUDA-graph Replay
        bypasses AOTI's ``run()``, so probe ops never fire. The runtime
        will hard-fail at ``init`` if both are set for the same method.
    """
    return w.clone()


@register_fake(_OP_QUALNAME)
def _probe_fake(w: torch.Tensor, probe_id: int) -> torch.Tensor:
    # Fresh fake tensor so inductor doesn't decide to inline the op away.
    return torch.empty_like(w)


PROBE_OP_TARGET = torch.ops.executorch_weight_offload.probe.default


# Payload field names. INTERNAL design intent for the partition-payload
# (or NamedDataStore -- see the "payload transport" open item in the
# module docstring) that ``CudaBackend.preprocess`` would write and
# ``cuda_backend.cpp::init`` would parse once wired. Names are
# namespaced by method so prefill and decode each get their own payload
# in the same .pte.
PAYLOAD_KEY_VERSION = "version"
PAYLOAD_KEY_METHOD_NAME = "method_name"
PAYLOAD_KEY_SCHEDULE = "schedule"
PAYLOAD_KEY_FLOOR = "floor_bytes"
PAYLOAD_KEY_PIN_FQNS = "pin_fqns"
# Per-FQN metadata block introduced in v2. List of dicts, one per
# unique(schedule) FQN, each carrying the shape/dtype/device the
# runtime needs to wrap borrowed pool allocations and validate the
# AOTI catalog. See `_serialize_payload` for the wire layout.
PAYLOAD_KEY_CONSTANTS_METADATA = "constants_metadata"

# Schema version for the emitted offload payload.
#
# v2 carries the per-FQN ``constants_metadata`` block (dtype / shape /
# strides / device). Without it the runtime can't reconstruct the
# original tensor metadata after installing dummies as the AOTI
# container's constants (extract_constants_map after install returns
# the dummies' placeholder metadata, not the originals). The block
# also enables the source-blob offset formula used to slice
# `_weights_blob` per-FQN without ever loading the constants to GPU.
# v1 is rejected at parse with a "rebuild required" message —
# maintaining two runtime paths for an experimental feature is dead
# weight.
SCHEMA_VERSION = 2


# --------------------------------------------------------------------------
# Private compile-spec keys and NamedData wire format
# --------------------------------------------------------------------------
# EXPERIMENTAL. All keys below carry leading underscores so they
# read as "internal" at every callsite and stay invisible to anyone who
# only inspects the public surface. End users opt in through the public
# ``CudaPartitioner(weight_offload=True, weight_offload_pin_fqns=[...])``
# kwargs, which translate to these internal COMPILE specs. The keys
# themselves stay internal so the stack's own callers can still build
# raw compile specs.
#
# Note the separate axis: load-time budget control. The PUBLIC RUNTIME
# spec is ``weight_offload_budget_mb`` (int megabytes); the INTERNAL
# RUNTIME spec for exact-byte budgets is
# ``_weight_offload_internal_budget_bytes`` (decimal string, used by
# tests that need byte-level precision below 1 MB granularity). Both
# are runtime specs (consumed via ``BackendInitContext::get_runtime_spec``),
# not compile specs.

# Compile-spec key that flips the entire offload pipeline on for a method:
# triggers the pass at preprocess time and tells the runtime to skip
# ``update_constants_from_blob``, install pre-load dummies, build the
# pinned host mirror from ``_weights_blob``, and serve probes through
# the bounded pool. Value: ``b"1"`` (any non-empty byte string treated
# as enabled; the runtime only checks presence).
COMPILE_SPEC_KEY_ENABLE = "_weight_offload_internal_enable"

# Compile-spec key carrying the pin set. Value: NUL-separated UTF-8
# FQNs. The pass validates the list against the catalog and includes
# it verbatim in the payload.
COMPILE_SPEC_KEY_PIN_FQNS = "_weight_offload_internal_pin_fqns"

# NamedDataStore key prefix; full key per method is
# ``f"{prefix}__{method_name}"`` so prefill and decode coexist in the
# same .pte.
NAMED_DATA_KEY_PREFIX = "_weight_offload_payload"


def named_data_key_for_method(method_name: str) -> str:
    """The NamedDataStore key the runtime looks up for ``method_name``'s
    offload payload."""
    return f"{NAMED_DATA_KEY_PREFIX}__{method_name}"


# Magic constant guarding the binary payload header. Spells out "ETWO"
# in little-endian ASCII — first four bytes of every payload, easy to
# grep in hex dumps.
PAYLOAD_MAGIC = 0x4F575445

# Bounded sizes that the runtime parser also enforces. Anything beyond
# these caps is a hard fail on both sides — a payload that big either
# means schema drift or an attacker-controlled artifact, neither of
# which we want to silently accept.
_MAX_STR_LEN = 1024
_MAX_SCHEDULE_COUNT = 1_000_000
_MAX_PIN_COUNT = 100_000
_MAX_CONSTANTS_METADATA_COUNT = 1_000_000
_MAX_NDIM = 32

# Single-device assumption. ``create_with_device`` is device-0-only
# today; lifting it requires plumbing a per-method device index
# through the payload, runtime, and partitioner kwargs.
_CUDA_DEVICE_TYPE = 1
_REQUIRED_DEVICE_INDEX = 0

# torch.dtype -> integer code, matching c10::ScalarType (which AOTI and
# the slim ScalarType the runtime uses both follow). The runtime's
# elementSize(static_cast<ScalarType>(dtype)) must agree with this.
#
# RESTRICTED to dtypes the slim ScalarType enum
# (backends/aoti/slim/c10/core/ScalarType.h) actually implements. The
# slim ScalarType is the subset the runtime can validate + size, so
# advertising dtype codes the runtime would ET_CHECK on (e.g. Double,
# Complex*, QInt*) is worse than refusing the export here. To add a
# dtype: (1) extend slim's ScalarType enum + elementSize switch +
# isValidScalarType, then (2) add the torch dtype here.
_TORCH_DTYPE_TO_C10 = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def _pack_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    if len(b) > _MAX_STR_LEN:
        raise ValueError(
            f"weight offload: string too long for payload ({len(b)} > "
            f"{_MAX_STR_LEN}): {s!r}"
        )
    buf += struct.pack("<I", len(b))
    buf += b


def _serialize_payload(payload: dict) -> bytes:
    """Serialize the offload payload to the runtime wire format.

    Layout (little-endian throughout):
      ``magic``                    uint32   = ``PAYLOAD_MAGIC``
      ``schema_version``           uint32   = 2
      ``method_name_len``          uint32
      ``method_name``              UTF-8 bytes
      ``floor_bytes``              uint64
      ``schedule_count``           uint32
      for each schedule entry:
        ``fqn_len``                uint32
        ``fqn``                    UTF-8 bytes
      ``pin_count``                uint32
      for each pin entry:
        ``fqn_len``                uint32
        ``fqn``                    UTF-8 bytes
      ``constants_metadata_count`` uint32
      for each metadata entry:
        ``fqn_len``                uint32
        ``fqn``                    UTF-8 bytes
        ``dtype``                  int32   (c10::ScalarType)
        ``ndim``                   uint32
        ``sizes[ndim]``            int64 each
        ``strides[ndim]``          int64 each
        ``storage_offset``         int64
        ``nbytes``                 uint64
        ``device_type``            int32   (== 1 = CUDA)
        ``device_index``           int32   (== 0; single-device only)

    Custom binary (not JSON) because the runtime must stay free of
    JSON-parser dependencies for a private, fixed-shape payload.
    """
    method_name = payload[PAYLOAD_KEY_METHOD_NAME]
    schedule = payload[PAYLOAD_KEY_SCHEDULE]
    pin_fqns = payload[PAYLOAD_KEY_PIN_FQNS]
    floor_bytes = payload[PAYLOAD_KEY_FLOOR]
    schema_version = payload[PAYLOAD_KEY_VERSION]
    constants_metadata = payload[PAYLOAD_KEY_CONSTANTS_METADATA]

    if len(schedule) > _MAX_SCHEDULE_COUNT:
        raise ValueError(
            f"weight offload: schedule too long ({len(schedule)} > "
            f"{_MAX_SCHEDULE_COUNT})"
        )
    if len(pin_fqns) > _MAX_PIN_COUNT:
        raise ValueError(
            f"weight offload: pin_fqns too long ({len(pin_fqns)} > "
            f"{_MAX_PIN_COUNT})"
        )
    if len(constants_metadata) > _MAX_CONSTANTS_METADATA_COUNT:
        raise ValueError(
            f"weight offload: constants_metadata too long "
            f"({len(constants_metadata)} > {_MAX_CONSTANTS_METADATA_COUNT})"
        )

    buf = bytearray()
    buf += struct.pack("<I", PAYLOAD_MAGIC)
    buf += struct.pack("<I", schema_version)
    _pack_str(buf, method_name)
    buf += struct.pack("<Q", floor_bytes)
    buf += struct.pack("<I", len(schedule))
    for fqn in schedule:
        _pack_str(buf, fqn)
    buf += struct.pack("<I", len(pin_fqns))
    for fqn in pin_fqns:
        _pack_str(buf, fqn)
    buf += struct.pack("<I", len(constants_metadata))
    for entry in constants_metadata:
        _pack_str(buf, entry["fqn"])
        buf += struct.pack("<i", entry["dtype"])
        sizes = entry["sizes"]
        strides = entry["strides"]
        if len(sizes) != len(strides):
            raise ValueError(
                f"weight offload: sizes/strides length mismatch for "
                f"{entry['fqn']!r}: {len(sizes)} vs {len(strides)}"
            )
        if len(sizes) > _MAX_NDIM:
            raise ValueError(
                f"weight offload: ndim {len(sizes)} > {_MAX_NDIM} for "
                f"{entry['fqn']!r}"
            )
        buf += struct.pack("<I", len(sizes))
        for s in sizes:
            buf += struct.pack("<q", s)
        for st in strides:
            buf += struct.pack("<q", st)
        buf += struct.pack("<q", entry["storage_offset"])
        buf += struct.pack("<Q", entry["nbytes"])
        buf += struct.pack("<i", entry["device_type"])
        buf += struct.pack("<i", entry["device_index"])
    return bytes(buf)


def _parse_pin_fqns_spec(value: bytes) -> list[str]:
    """Decode the NUL-separated UTF-8 pin-FQN compile-spec value into a
    list. Empty value -> empty list."""
    if not value:
        return []
    return [chunk.decode("utf-8") for chunk in value.split(b"\x00") if chunk]


def pin_fqns_from_specs(compile_specs) -> list[str]:
    """Extract pin-FQN list from compile specs (NUL-separated UTF-8).
    Empty list if the spec is absent."""
    for spec in compile_specs:
        if spec.key == COMPILE_SPEC_KEY_PIN_FQNS:
            return _parse_pin_fqns_spec(bytes(spec.value))
    return []


def _placeholder_fqn_map(exported_program) -> dict[str, str]:
    """Build a ``placeholder_node.name -> FQN`` map for parameter and
    buffer placeholders. Other placeholder kinds are intentionally
    skipped — USER_INPUT is per-call data, CONSTANT_TENSOR is too
    fine-grained for offload's bookkeeping, and the rest aren't
    tensor weights.

    ``torch.export`` auto-lifts inline scalar literals (e.g. the
    ``1.0`` in ``x + 1.0``) into buffer placeholders with the name
    ``_lifted_tensor_constant*``. AOTI folds these into kernel code
    and never surfaces them through ``extract_constants_map``, so
    probing them would hand the runtime a schedule FQN that fails
    the ``schedule ⊆ catalog`` check at init. Skip them by name —
    matches the torch.export naming convention; if upstream renames
    the prefix, the runtime catalog check will catch the drift.
    """
    from torch.export.graph_signature import InputKind

    return {
        spec.arg.name: spec.target
        for spec in exported_program.graph_signature.input_specs
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
        and spec.target is not None
        and not spec.target.startswith("_lifted_tensor_constant")
    }


def _is_view_op(target) -> bool:
    """True iff ``target`` is a read-only view op (return aliases an
    input; no in-place mutation on any input). The pass treats view
    ops as transparent metadata operations — it never probes them
    directly; it duplicates them per non-view consumer so each
    consumer sees its own ``probe(root, probe_id) -> view_chain``
    pipeline."""
    if not isinstance(target, torch._ops.OpOverload):
        return False
    schema = target._schema
    if not any(ret.alias_info is not None for ret in schema.returns):
        return False
    return not any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in schema.arguments
    )


_warned_view_targets: set = set()


def _warn_unrecognised_view_once(node) -> None:
    """Hint when ``_build_alias_chains`` couldn't pick a source for a
    view-op node. Dedup keyed on op target so the same custom view
    appearing N times emits one warning, not N."""
    target = getattr(node, "target", None)
    if target in _warned_view_targets:
        return
    _warned_view_targets.add(target)
    sys.stderr.write(
        f"[weight_offload_pass] view op {target!r} (node {node.name!r}) "
        f"had no input matching a known placeholder or alias chain; "
        f"this view will NOT be probed. If you registered a custom "
        f"view op that aliases an input not at a leading position, "
        f"or wraps the aliased input through an unrecognised pattern, "
        f"the offload pass cannot trace the alias and weights backing "
        f"that view will fall back to AOTI's eager constants.\n"
    )


def _build_alias_chains(graph, placeholder_fqn):
    """Map view-chain Nodes back to their root placeholder + the chain
    of view ops that produced them.

    Returns ``alias_info: dict[Node, (root_placeholder, [view_node, ...])]``.
    Applying the chain in order, starting from ``root_placeholder``,
    reproduces the alias node — that's what
    ``_duplicate_view_chain`` re-emits on top of each fresh probe.
    """
    alias_info: dict[torch.fx.Node, tuple[torch.fx.Node, list[torch.fx.Node]]] = {}
    for node in graph.nodes:
        if node.op != "call_function" or not _is_view_op(node.target):
            continue
        # Views can have multiple tensor inputs (e.g.
        # ``expand_as(self, other)`` aliases ``self`` only). We pick
        # the FIRST input that's either a placeholder or an
        # already-recognised alias. ATen view ops universally place
        # the aliased tensor at the leading position, but we don't
        # hard-code ``all_input_nodes[0]`` because that would silently
        # mis-handle any custom view op that places its aliased input
        # elsewhere — a first-match scan tolerates the convention
        # without committing to it.
        source = next(
            (
                inp
                for inp in node.all_input_nodes
                if inp.name in placeholder_fqn or inp in alias_info
            ),
            None,
        )
        if source is None:
            # Diagnostic: a view op whose inputs include something
            # that LOOKS like it should alias a placeholder
            # (transitively) but doesn't match our placeholder /
            # already-aliased set. Most commonly this is benign (a
            # view on a fresh intermediate result), but it can also
            # mean a custom view op rewrote the aliased input through
            # an unrecognised wrapper. Emit a one-time hint so the
            # case isn't silently dropped without leaving a trace.
            _warn_unrecognised_view_once(node)
            continue
        if source.name in placeholder_fqn:
            alias_info[node] = (source, [node])
        else:
            root, chain = alias_info[source]
            alias_info[node] = (root, [*chain, node])
    return alias_info


def _duplicate_view_chain(graph, probe_node, chain, original_root):
    """Re-emit each view op in ``chain`` rooted at ``probe_node`` instead
    of ``original_root``. Returns the new tail node (the leaf of the
    duplicated chain — what the consumer will read from)."""
    current = probe_node
    prev_original = original_root
    for orig in chain:
        new_args = tuple(current if a is prev_original else a for a in orig.args)
        new_node = graph.call_function(
            orig.target,
            args=new_args,
            kwargs=orig.kwargs,
        )
        if "val" in orig.meta:
            new_node.meta["val"] = orig.meta["val"]
        prev_original = orig
        current = new_node
    return current


def _insert_probes_for_node(
    node, graph, placeholder_fqn, alias_info, schedule, probe_fqn
):
    """Walk ``node.args`` / ``node.kwargs``, replacing each placeholder
    or view-of-placeholder reference with a freshly-inserted
    ``probe(root, probe_id)`` followed (when the original arg was a
    view) by a duplicated view chain. Mutates ``schedule`` by
    appending the rewritten FQN at the new probe's ``probe_id``, and
    ``probe_fqn`` with the inserted probe Node mapped to its FQN
    (used downstream by the transitive floor analysis)."""

    def _maybe_probe(arg):
        if not isinstance(arg, torch.fx.Node):
            return arg

        if arg.name in placeholder_fqn:
            root = arg
            chain: list[torch.fx.Node] = []
        elif arg in alias_info:
            root, chain = alias_info[arg]
        else:
            return arg

        fqn = placeholder_fqn[root.name]
        probe_id = len(schedule)
        schedule.append(fqn)
        with graph.inserting_before(node):
            probe_node = graph.call_function(
                PROBE_OP_TARGET,
                args=(root, probe_id),
            )
            if "val" in root.meta:
                probe_node.meta["val"] = root.meta["val"]
            probe_fqn[probe_node] = fqn
            return _duplicate_view_chain(graph, probe_node, chain, root)

    node.args = torch.fx.map_arg(node.args, _maybe_probe)
    node.kwargs = torch.fx.map_arg(node.kwargs, _maybe_probe)


def _fusion_dependency_sets(
    graph, probe_fqn: dict[torch.fx.Node, str]
) -> tuple[dict[torch.fx.Node, set[str]], dict[torch.fx.Node, set[str]]]:
    """Per-node FX fusion model for the floor calculation.

    Returns ``(working_sets, output_deps)`` after a single topo walk:

      probe node           output_deps = {fqn}
      view op              output_deps = union(upstream output_deps)
      fusible call         working_set = union(upstream output_deps)
                           output_deps = working_set
      materializing barrier
                           working_set = union(upstream output_deps)
                           output_deps = {}

    No op is treated as a confirmed materializing barrier today --
    proving an op cannot fuse with downstream users under
    AOTI/Inductor needs the post-lowering kernel-grouping work that
    a future commit will do. Defaulting to "fusible" overestimates
    the floor (safe); claiming barrier where none exists
    underestimates it (corruption). See the ``floor_bytes``
    description in ``_apply_weight_offload`` for the explicit
    upper-bound contract.
    """
    working_sets: dict[torch.fx.Node, set[str]] = {}
    output_deps: dict[torch.fx.Node, set[str]] = {}

    for node in graph.nodes:
        upstream: set[str] = set()
        for inp in node.all_input_nodes:
            up = output_deps.get(inp)
            if up:
                upstream |= up

        if node in probe_fqn:
            output_deps[node] = {probe_fqn[node]}
            continue

        if node.op == "output":
            # Output is a fusion sink: Inductor may fuse independent
            # final consumers into one multi-output kernel that
            # reads every weight any of them touched. Treat it as a
            # candidate so its upstream union shows up in the pair
            # window — without this the floor misses the case of
            # ``return x*w1, x*w2, x*w3, x*w4``.
            output_deps[node] = upstream
            if upstream:
                working_sets[node] = upstream
            continue

        if node.op != "call_function":
            # Placeholders, get_attr: nothing to fuse, just carry
            # whatever came in (empty for true leaves).
            output_deps[node] = upstream
            continue

        if _is_view_op(node.target):
            # Transparent metadata op — no kernel, no working set.
            output_deps[node] = upstream
            continue

        # Default: fusible / non-materializing. Replace this branch
        # with an ``output_deps[node] = set()`` reset once a
        # confirmed barrier list lands.
        working_sets[node] = upstream
        output_deps[node] = upstream

    return working_sets, output_deps


def _compute_floor_bytes(
    graph,
    probe_fqn: dict[torch.fx.Node, str],
    pin_set: set[str],
    nbytes_of,
) -> int:
    """Conservative FX fusion upper bound on the streaming-pool
    floor — NOT a tight kernel-level estimate. See the contract
    text in ``_apply_weight_offload``.

    Formula: ``max over consecutive FX candidate pairs of (sum
    bytes of the UNION of non-pinned working sets at each side) +
    max single non-pinned weight``. FX candidates are non-view
    non-probe ``call_function`` nodes plus the output sink (the
    output is included so that Inductor fusing independent final
    consumers into one multi-output kernel still factors in). The
    working set comes from ``_fusion_dependency_sets`` and
    propagates probe FQNs through every fusion-eligible edge (the
    pass refuses to claim any op is a barrier without proof). The
    pair window layers depth-1 prefetch on top; the ``+ max
    single`` is eviction headroom for swapping the largest weight.
    """
    working_sets, _ = _fusion_dependency_sets(graph, probe_fqn)

    def non_pinned(s: set[str]) -> set[str]:
        return s - pin_set

    # ``working_sets`` is populated in ``graph.nodes`` iteration
    # order, so its insertion order is graph order in Python 3.7+
    # — the pair window relies on that for "consecutive consumers".
    candidates = [
        (node, non_pinned(ws)) for node, ws in working_sets.items() if non_pinned(ws)
    ]
    if not candidates:
        return 0

    def sum_bytes(fqns: set[str]) -> int:
        return sum(nbytes_of(fqn) for fqn in fqns)

    n = len(candidates)
    max_window = max(
        sum_bytes(candidates[i][1] | (candidates[i + 1][1] if i + 1 < n else set()))
        for i in range(n)
    )
    all_non_pinned: set[str] = set().union(*(ws for _, ws in candidates))
    max_single = max(nbytes_of(fqn) for fqn in all_non_pinned)
    return max_window + max_single


def _apply_weight_offload(
    exported_program,
    *,
    method_name: str,
    pin_fqns: list[str] | None = None,
) -> dict:
    """In-place graph rewrite + offload payload computation.

    Internal: the only supported caller is
    ``CudaBackend.pre_aoti_transform_and_collect_named_data``. The
    ``method_name`` arg is required (no default) so a direct caller
    on a multi-method model cannot silently collide all methods on
    ``"forward"``.

    Mutates ``exported_program`` in place: inserts ``probe(w, probe_id)``
    in front of every consumer of every parameter / buffer placeholder
    and rewrites the consumer's arg to read the probe's output. One
    probe per ``(consumer, weight)`` pair so the runtime can re-load
    a weight evicted between two uses in the same forward pass.
    ``probe_id`` is dense 0..N-1 in graph order.

    Returns the offload payload dict (see ``PAYLOAD_KEY_*`` at module
    scope for the schema):
      * ``schedule[probe_id]`` is the FQN that probe site reads;
        pinned FQNs appear here too (the runtime picks the resident
        path inside ``serve``).
      * ``floor_bytes`` is a conservative FX-fusion-aware upper bound
        on the streaming pool, excluding pinned FQNs. The runtime
        hard-fails if ``budget - pinned < floor``.
      * ``pin_fqns`` is the resident set, deduped first-seen-order.
      * ``constants_metadata`` carries per-FQN dtype / sizes / strides
        / storage_offset / nbytes / device for runtime cross-check.

    Re-entry: this pass MUST run before AOTI compile (it operates on
    placeholders) and MUST NOT be re-run on a graph that already
    contains probe nodes -- the second pass would insert probes on
    the probes' outputs.
    """
    pin_fqns = list(pin_fqns or [])
    graph = exported_program.graph_module.graph

    # Re-entering the pass would wrap each probe's input (still a
    # placeholder Node) in a fresh probe, leaving the graph with
    # probe-of-probe-of-w and a schedule that no longer matches the
    # ``probe_id`` values the runtime sees. Loud failure beats subtle
    # wrong answers.
    existing = sum(
        1
        for n in graph.nodes
        if n.op == "call_function" and n.target is PROBE_OP_TARGET
    )
    if existing:
        raise RuntimeError(
            f"weight offload: pass already applied to this ExportedProgram "
            f"({existing} probe(s) present); re-applying is a caller bug"
        )

    placeholder_fqn = _placeholder_fqn_map(exported_program)
    catalog_fqns = set(placeholder_fqn.values())
    for fqn in pin_fqns:
        if fqn not in catalog_fqns:
            raise ValueError(
                f"weight offload: pin_fqns references {fqn!r} which is not a "
                f"parameter or buffer placeholder; available: "
                f"{sorted(catalog_fqns)}"
            )

    # Snapshot the original view chains before any insertion. The
    # main loop's iteration is also a snapshot, so newly-inserted
    # probe and duplicated-view nodes are NOT re-traversed.
    alias_info = _build_alias_chains(graph, placeholder_fqn)

    # Walk consumers in graph order. View ops are skipped — they get
    # duplicated per non-view consumer inside ``_insert_probes_for_node``.
    # ``probe_id`` is assigned by ``len(schedule)`` at insertion time,
    # which gives contiguous 0..N-1 in the order ``torch.fx.map_arg``
    # visits non-view consumer arguments — the same order ``schedule``
    # indexes, which is what the runtime ultimately keys on.
    schedule: list[str] = []
    probe_fqn: dict[torch.fx.Node, str] = {}
    for node in list(graph.nodes):
        if node.op != "call_function" or _is_view_op(node.target):
            continue
        _insert_probes_for_node(
            node, graph, placeholder_fqn, alias_info, schedule, probe_fqn
        )

    # Original view chains lose their last user once consumers are
    # rewritten onto duplicates. Drop them so the graph stays small
    # and AOTI doesn't waste codegen on unreferenced views.
    graph.eliminate_dead_code()
    exported_program.graph_module.recompile()

    state_dict = exported_program.state_dict
    constants = getattr(exported_program, "constants", {}) or {}

    def _nbytes(fqn: str) -> int:
        if fqn in state_dict:
            t = state_dict[fqn]
        elif fqn in constants:
            t = constants[fqn]
        else:
            raise KeyError(
                f"weight offload: FQN {fqn!r} appears as a placeholder but "
                f"is missing from state_dict and constants"
            )
        # The host mirror is sized at numel * element_size; a non-
        # contiguous tensor would over-read its storage. The runtime
        # parser also rejects non-contiguous metadata, but flagging
        # here names the FQN with a Python stack trace.
        if not t.is_contiguous():
            raise ValueError(
                f"weight offload: FQN {fqn!r} is non-contiguous "
                f"(shape={tuple(t.shape)}, strides={tuple(t.stride())}); "
                f"call .contiguous() on the source tensor before exporting"
            )
        return t.numel() * t.element_size()

    pin_set = set(pin_fqns)
    # Drop probes from the map whose nodes were eliminated as dead
    # code (defensive — the rewriter currently always leaves probes
    # with a consumer).
    live_nodes = set(graph.nodes)
    probe_fqn = {p: f for p, f in probe_fqn.items() if p in live_nodes}
    floor_bytes = _compute_floor_bytes(graph, probe_fqn, pin_set, _nbytes)

    # Build per-FQN metadata for unique(schedule). v2 payload requires
    # this so the runtime can wrap pool allocations with original
    # dtype/shape after replacing AOTI's constants with dummies.
    constants_metadata = _build_constants_metadata(schedule, state_dict, constants)

    return {
        PAYLOAD_KEY_VERSION: SCHEMA_VERSION,
        PAYLOAD_KEY_METHOD_NAME: method_name,
        PAYLOAD_KEY_SCHEDULE: schedule,
        PAYLOAD_KEY_FLOOR: floor_bytes,
        PAYLOAD_KEY_PIN_FQNS: pin_fqns,
        PAYLOAD_KEY_CONSTANTS_METADATA: constants_metadata,
    }


def _build_constants_metadata(
    schedule: list[str], state_dict: dict, constants: dict
) -> list[dict]:
    """One entry per unique(schedule) FQN, preserving first-seen order
    so the runtime can iterate deterministically. The runtime parser
    re-validates set-equality with unique(schedule); first-seen order
    here matches the order the pass emits, which keeps any later
    diff-based debugging readable."""
    out: list[dict] = []
    seen: set[str] = set()
    for fqn in schedule:
        if fqn in seen:
            continue
        seen.add(fqn)
        if fqn in state_dict:
            t = state_dict[fqn]
        elif fqn in constants:
            t = constants[fqn]
        else:
            raise KeyError(
                f"weight offload: FQN {fqn!r} appears in schedule but "
                f"is missing from state_dict and constants"
            )
        if t.dtype not in _TORCH_DTYPE_TO_C10:
            raise ValueError(
                f"weight offload: FQN {fqn!r} has unsupported dtype "
                f"{t.dtype}; extend _TORCH_DTYPE_TO_C10 and the runtime "
                f"ScalarType enum together"
            )
        # Offload mirrors each constant's LOGICAL bytes densely. A constant
        # backed by a storage larger than its logical size (a view into a
        # bigger buffer: contiguous + offset 0 but backed by extra bytes,
        # e.g. ``big[:k]``) would break that: AOTI's per-constant data_size
        # is ``untyped_storage().nbytes()``, so the dense source-blob offset
        # math the runtime relies on would misalign. The runtime catches
        # this via its nbytes==data_size cross-check; naming the FQN at
        # export is friendlier. The runtime parser stays the trust boundary
        # (including the cuda:0 device requirement, which is enforced there
        # and re-checked by AOTI's user-managed install); this is UX only.
        if t.untyped_storage().nbytes() != t.numel() * t.element_size():
            raise ValueError(
                f"weight offload: FQN {fqn!r} is backed by a storage "
                f"({t.untyped_storage().nbytes()} bytes) larger than its "
                f"logical size ({t.numel() * t.element_size()} bytes); it "
                f"likely aliases a view into a bigger buffer. Call "
                f".contiguous().clone() on the source tensor before exporting."
            )
        sizes = list(t.shape)
        # Hard-fail zero-size constants at the pass. cudaHostAlloc(0)
        # and friends are undefined; better to surface this at export
        # than to ship four under-tested edge cases.
        product = 1
        for s in sizes:
            product *= s
        if product == 0:
            raise ValueError(
                f"weight offload: FQN {fqn!r} has zero-product shape "
                f"{sizes}; offload does not support empty constants"
            )
        out.append(
            {
                "fqn": fqn,
                "dtype": _TORCH_DTYPE_TO_C10[t.dtype],
                "sizes": sizes,
                "strides": list(t.stride()),
                "storage_offset": int(t.storage_offset()),
                "nbytes": t.numel() * t.element_size(),
                "device_type": _CUDA_DEVICE_TYPE,
                "device_index": _REQUIRED_DEVICE_INDEX,
            }
        )
    return out
