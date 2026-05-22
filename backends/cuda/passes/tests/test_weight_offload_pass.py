# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for ``_apply_weight_offload``: payload contents,
probe-node placement, probe_id ↔ schedule alignment, view-chain
duplication, floor set-union semantics, and hard-fail paths. Runtime
serve and partitioner plumbing are covered by other test files."""

import unittest

import torch
import torch.nn as nn
from executorch.backends.cuda.passes.weight_offload_pass import (
    _apply_weight_offload,
    PAYLOAD_KEY_FLOOR,
    PAYLOAD_KEY_METHOD_NAME,
    PAYLOAD_KEY_PIN_FQNS,
    PAYLOAD_KEY_SCHEDULE,
    PAYLOAD_KEY_VERSION,
    PROBE_OP_TARGET,
    SCHEMA_VERSION,
)
from torch.export import export


class _SingleConsumerModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        return x @ self.w


class _MultiConsumerModel(nn.Module):
    """One weight, two consumers — exercises the dense ``probe_id`` rule."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        return (x @ self.w) + ((x + 1) @ self.w)


class _TwoWeightModel(nn.Module):
    """Two distinct weights, used in sequence — exercises floor arithmetic.

    Sizes are intentionally asymmetric so ``max_single`` differs from
    ``max_pair`` in the floor formula:
      - ``w1``: 8x8 float32  -> 256 bytes
      - ``w2``: 16x8 float32 -> 512 bytes
    """

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(8, 8))
        self.w2 = nn.Parameter(torch.randn(16, 8))

    def forward(self, x):
        # x: [4, 8]
        y = x @ self.w1  # [4,8] @ [8,8] -> [4,8]
        return y @ self.w2.T  # [4,8] @ [8,16] -> [4,16]


class _BufferModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.register_buffer("buf", torch.randn(dim, dim))

    def forward(self, x):
        return x @ self.buf


class _ViewChainModel(nn.Module):
    """One weight viewed via .T and consumed by two kernels.

    Exercises the view-chain duplication path: the pass must emit two
    probes (one per kernel) rooted at the placeholder, each with its
    own duplicated transpose, so a later kernel can re-load ``w`` if
    the runtime evicts it between the two reads.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        v = self.w.T  # single shared view
        return (x @ v) + ((x + 1) @ v)


class _IndependentMultiOutputModel(nn.Module):
    """``return x*w1, x*w2, x*w3, x*w4`` — independent pointwise
    branches that Inductor can fuse into a single multi-output
    kernel reading all four weights. The output node is the
    fusion sink that ties them together at floor-calculation time.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim, dim))
        self.w3 = nn.Parameter(torch.randn(dim, dim))
        self.w4 = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        return x * self.w1, x * self.w2, x * self.w3, x * self.w4


class _SequentialMatmulModel(nn.Module):
    """``y = x @ w1; z = y @ w2`` — two matmuls in series.

    Documents the current policy: matmul is NOT treated as a
    confirmed materializing barrier, so the second matmul's working
    set includes BOTH weights even though it physically only reads
    w2. That is a conservative overestimate — safe under the
    "if unsure, propagate" rule. Could be tightened by adding matmul
    to a verified barrier list once post-AOTI kernel grouping data
    is available.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        y = x @ self.w1
        return y @ self.w2


class _FusablePointwiseChainModel(nn.Module):
    """Pointwise chain Inductor can fuse into one kernel reading all
    four weights simultaneously. The pre-AOTI floor analysis must
    upper-bound this by walking the transitive probe cone — the
    direct-deps-only formula would size the pool for ~2 weights and
    let the runtime evict a weight the fused kernel still needs."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim, dim))
        self.w3 = nn.Parameter(torch.randn(dim, dim))
        self.w4 = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        return x * self.w1 + self.w2 + self.w3 + self.w4


class _SharedWeightKernelModel(nn.Module):
    """Two kernels share one weight directly (no view) — the floor's
    set-union semantics must collapse them so the shared weight isn't
    double-counted across the pair."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(8, 8))  # 256 bytes
        self.w2 = nn.Parameter(torch.randn(8, 8))  # 256 bytes

    def forward(self, x):
        a = x @ self.w
        b = (x + 1) @ self.w  # shares w with `a`
        c = a @ self.w2
        return b + c


def _export(model: nn.Module, *example_inputs) -> "torch.export.ExportedProgram":
    return export(model, example_inputs, strict=True)


def _probe_nodes(ep) -> list[torch.fx.Node]:
    return [
        n
        for n in ep.graph_module.graph.nodes
        if n.op == "call_function" and n.target == PROBE_OP_TARGET
    ]


class TestApplyWeightOffload(unittest.TestCase):
    # -----------------------------------------------------------------
    # Probe insertion
    # -----------------------------------------------------------------

    def test_single_consumer_inserts_one_probe(self):
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        probes = _probe_nodes(ep)
        self.assertEqual(len(probes), 1)
        self.assertEqual(payload[PAYLOAD_KEY_SCHEDULE], ["w"])

    def test_multi_consumer_inserts_one_probe_per_consumer(self):
        ep = _export(_MultiConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        probes = _probe_nodes(ep)
        self.assertEqual(len(probes), 2)
        # Same FQN appears twice — once per consumer site.
        self.assertEqual(payload[PAYLOAD_KEY_SCHEDULE], ["w", "w"])

    def test_buffer_placeholder_is_probed(self):
        ep = _export(_BufferModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        self.assertEqual(len(_probe_nodes(ep)), 1)
        self.assertEqual(payload[PAYLOAD_KEY_SCHEDULE], ["buf"])

    def test_view_chain_inserts_probe_per_kernel(self):
        """A weight viewed via ``.T`` and consumed by two kernels gets
        two probes — one per kernel, each rooted at the placeholder
        with its own duplicated view chain — so a later kernel can
        re-load the weight if the runtime evicted it between the
        reads. The original shared view is dead-code-eliminated."""
        ep = _export(_ViewChainModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        probes = _probe_nodes(ep)
        self.assertEqual(len(probes), 2)
        # Both probes wrap the placeholder ``w`` directly, NOT the
        # transpose's output — the c-shim needs to see the original
        # weight pointer to resolve the FQN via ProbeRegistry.
        sig = ep.graph_signature
        placeholder_fqn = {
            spec.arg.name: spec.target
            for spec in sig.input_specs
            if spec.target is not None
        }
        for probe in probes:
            self.assertIn(probe.args[0].name, placeholder_fqn)
            self.assertEqual(placeholder_fqn[probe.args[0].name], "w")
        self.assertEqual(payload[PAYLOAD_KEY_SCHEDULE], ["w", "w"])

    def test_probe_ids_contiguous_from_zero(self):
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        _apply_weight_offload(ep, method_name="forward")
        probes = _probe_nodes(ep)
        ids = [n.args[1] for n in probes]
        self.assertEqual(ids, list(range(len(probes))))

    def test_schedule_indexed_by_probe_id(self):
        """``schedule[probe_id]`` matches the FQN the probe wraps."""
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        probes = _probe_nodes(ep)
        # The probe's first arg is the placeholder it wraps; resolve
        # back to FQN via the graph signature.
        sig = ep.graph_signature
        placeholder_fqn = {
            spec.arg.name: spec.target
            for spec in sig.input_specs
            if spec.target is not None
        }
        for probe in probes:
            probe_id = probe.args[1]
            wrapped_fqn = placeholder_fqn[probe.args[0].name]
            self.assertEqual(payload[PAYLOAD_KEY_SCHEDULE][probe_id], wrapped_fqn)

    def test_re_entry_is_hard_error(self):
        """Pass is single-shot; re-applying would wrap probes in probes
        and silently corrupt the probe_id -> FQN mapping, so the second
        call raises instead."""
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        _apply_weight_offload(ep, method_name="forward")
        with self.assertRaises(RuntimeError) as cm:
            _apply_weight_offload(ep, method_name="forward")
        self.assertIn("already applied", str(cm.exception))

    # -----------------------------------------------------------------
    # Payload schema
    # -----------------------------------------------------------------

    def test_payload_has_v2_schema_version(self):
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        self.assertEqual(payload[PAYLOAD_KEY_VERSION], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 2)

    def test_payload_echoes_method_name(self):
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="prefill")
        self.assertEqual(payload[PAYLOAD_KEY_METHOD_NAME], "prefill")

    def test_payload_echoes_pin_fqns(self):
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward", pin_fqns=["w1"])
        self.assertEqual(payload[PAYLOAD_KEY_PIN_FQNS], ["w1"])

    def test_payload_pin_fqns_defaults_empty(self):
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        self.assertEqual(payload[PAYLOAD_KEY_PIN_FQNS], [])

    # -----------------------------------------------------------------
    # Floor arithmetic
    #
    # The floor is a conservative FX fusion upper bound, NOT a
    # tight kernel-level estimate. ``W_i`` here is the working set
    # at FX candidate ``i`` (a non-view non-probe ``call_function``
    # plus the output sink), built by propagating probe FQNs
    # forward through every fusion-eligible edge — the pass refuses
    # to claim any op is a barrier without proof. Formula:
    # ``max(sum bytes W_i ∪ W_{i+1}) + max single weight``. Set
    # union across the pair window collapses weights shared between
    # adjacent candidates (the pool only needs one allocation).
    # -----------------------------------------------------------------

    def test_floor_two_weights_unpinned(self):
        """Two kernels, each reads one distinct weight."""
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # working_sets = [{"w1"}, {"w2"}]; bytes(w1)=256, bytes(w2)=512.
        # max_window over (S_0 ∪ S_1)=768 and (S_1)=512 -> 768.
        # max_single = 512. floor = 1280.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 768 + 512)

    def test_floor_pinned_weight_excluded(self):
        """Pinning w1 collapses the streaming sequence to one kernel
        (the one reading w2); floor = max_window + max_single = 512+512."""
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward", pin_fqns=["w1"])
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 512 + 512)

    def test_floor_all_pinned_is_zero(self):
        ep = _export(_TwoWeightModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(
            ep, method_name="forward", pin_fqns=["w1", "w2"]
        )
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 0)

    def test_floor_single_consumer(self):
        """One weight, one consumer: max_window = sum(S_0) = bytes(w);
        max_single = bytes(w); floor = 2 * bytes(w)."""
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # bytes(w) = 8*8*4 = 256
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 256 + 256)

    def test_floor_shared_weight_uses_set_union(self):
        """Two kernels reading the same weight don't double-count it
        in the pair window — the pool only needs one allocation."""
        ep = _export(_MultiConsumerModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # working_sets = [{"w"}, {"w"}]; pair union -> {"w"} (256 bytes).
        # max_window = 256, max_single = 256, floor = 512.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 512)

    def test_floor_shared_weight_across_consecutive_kernels(self):
        """Three kernels: K_0 = {w}, K_1 = {w}, K_2 = {w, w2}. The
        largest pair window is K_1 ∪ K_2 = {w, w2}, so the pool must
        hold both bytes, not just the larger single."""
        ep = _export(_SharedWeightKernelModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # bytes(w)=256, bytes(w2)=256. max_window = 512 (K_1 ∪ K_2 or
        # K_2 alone). max_single = 256. floor = 768.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 768)

    def test_floor_sequential_matmul_documents_no_barrier_reset(self):
        """``y = x @ w1; z = y @ w2`` — the second matmul's working
        set includes BOTH w1 and w2 even though the second matmul
        physically only rereads w2. The pass refuses to claim
        matmul is a barrier under AOTI/Inductor without proof, so
        deps propagate. Conservative overestimate — safe — but
        documents the cost of "if unsure, propagate"."""
        ep = _export(_SequentialMatmulModel(), torch.randn(4, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # bytes(w1) = bytes(w2) = 256.
        # working_sets: {mm1: {w1}, mm2: {w1, w2}}
        # max_window = max(|mm1 ∪ mm2|, |mm2|) = 512.
        # max_single = 256. floor = 768.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 768)

    def test_floor_independent_multi_output_includes_all_branches(self):
        """``return x*w1, x*w2, x*w3, x*w4`` — independent pointwise
        branches Inductor can fuse into one multi-output kernel
        reading all four weights. Without treating the output node
        as a fusion sink, adjacent FX branches would each contribute
        only ~2 weights to any pair window, missing this case."""
        ep = _export(_IndependentMultiOutputModel(), torch.randn(8, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # bytes(each w) = 256. Output-sink working set = {w1, w2,
        # w3, w4} = 1024. max_window pair touching the output sink
        # >= 1024. max_single = 256. floor = 1280.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 1024 + 256)

    def test_floor_fusable_pointwise_chain_uses_transitive_cone(self):
        """``x * w1 + w2 + w3 + w4`` is a pointwise chain Inductor
        can fuse into one kernel. The transitive cone at the final
        add covers all four weights — the floor must size the pool
        for that, plus eviction headroom. Direct-deps-only sizing
        would admit ~2 weights and corrupt the fused kernel's read
        of weights 3 and 4."""
        ep = _export(_FusablePointwiseChainModel(), torch.randn(8, 8))
        payload = _apply_weight_offload(ep, method_name="forward")
        # Each weight = 8*8*4 = 256 bytes. Transitive cone at the
        # final add = {w1, w2, w3, w4} = 1024 bytes. max_single = 256.
        # floor = 1024 + 256 = 1280.
        self.assertEqual(payload[PAYLOAD_KEY_FLOOR], 1024 + 256)

    # -----------------------------------------------------------------
    # Hard fails
    # -----------------------------------------------------------------

    def test_unknown_pin_fqn_is_hard_error(self):
        ep = _export(_SingleConsumerModel(), torch.randn(4, 8))
        with self.assertRaises(ValueError) as cm:
            _apply_weight_offload(
                ep, method_name="forward", pin_fqns=["does_not_exist"]
            )
        self.assertIn("does_not_exist", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
