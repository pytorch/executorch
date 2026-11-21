# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import unittest

import torch

from executorch.backends.cuda.passes.lower_offgraph_kv import (
    _private_batched_sequence_step_fn,
)
from executorch.backends.cuda.triton.kernels.sdpa_varlen import sdpa_varlen


def _case(owners, previous, window, heads=2, dim=128, groups=4, padding=0):
    """Same logical K/V in legacy columns and packed private spans.

    Torch owns full declared storage here; NaNs mark every byte outside the
    runtime's private spans. Native pool tests verify the actual smaller CUDA
    allocations, which cannot be expressed by a bounds-checked Torch view.
    """
    torch.manual_seed(917)
    columns, max_write, context = 65, 1024, 2048
    width = len(owners) + padding
    counts = collections.Counter(owners)
    ends = {owner: previous.get(owner, 0) + count for owner, count in counts.items()}
    rows = min(context, window + max_write - 1) if window else context
    shape = (rows, columns, heads, dim)
    legacy_k = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    legacy_v = torch.zeros_like(legacy_k)
    private_k = torch.full_like(legacy_k, float("nan"))
    private_v = torch.full_like(legacy_v, float("nan"))
    layouts = torch.zeros((columns, 2), dtype=torch.int32, device="cuda")
    steps = torch.zeros((columns, 3), dtype=torch.int32, device="cuda")
    seq_cols = torch.full((max_write,), columns - 1, dtype=torch.int64, device="cuda")
    permutation = torch.arange(max_write, dtype=torch.int64, device="cuda")
    seq_info = torch.zeros((2 * (columns - 1), 4), dtype=torch.int32, device="cuda")
    k_step = torch.zeros((1, heads, width, dim), dtype=torch.bfloat16, device="cuda")
    v_step = torch.zeros_like(k_step)
    k_stage = torch.empty(
        (1, max_write, heads, dim), dtype=torch.bfloat16, device="cuda"
    )
    v_stage = torch.empty_like(k_stage)
    q = torch.randn(
        (1, heads * groups, width, dim), dtype=torch.bfloat16, device="cuda"
    )
    position = torch.zeros(width, dtype=torch.int64, device="cuda")
    payload, spans = {}, {}
    base = stage_start = 0
    for owner, end in sorted(ends.items()):
        capacity = min(window, end) if window else end
        spans[owner] = (base, capacity)
        layouts[owner] = torch.tensor(
            [base, capacity], dtype=torch.int32, device="cuda"
        )
        start = previous.get(owner, 0)
        steps[owner] = torch.tensor(
            [stage_start, start, counts[owner]], dtype=torch.int32, device="cuda"
        )
        payload[owner] = [
            torch.randn((end, heads, dim), dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        for values, legacy, private in zip(
            payload[owner], (legacy_k, legacy_v), (private_k, private_v)
        ):
            positions = torch.arange(max(0, end - rows), end, device="cuda")
            legacy[positions % rows, owner] = values[positions]
            old = torch.arange(max(0, start - capacity), start, device="cuda")
            physical = old % capacity if window else old
            private.reshape(-1, heads, dim)[base + physical] = values[old]
        base += capacity
        stage_start += counts[owner]
    layouts[-1] = torch.tensor([base, 1], dtype=torch.int32, device="cuda")
    cursor = dict(previous)
    stage_cursor = {owner: int(steps[owner, 0]) for owner in ends}
    segments = []
    for token, owner in enumerate(owners):
        absolute = cursor.get(owner, 0)
        seq_cols[token], position[token] = owner, absolute
        permutation[token] = stage_cursor[owner]
        k_step[0, :, token] = payload[owner][0][absolute]
        v_step[0, :, token] = payload[owner][1][absolute]
        if not segments or segments[-1][2] != owner:
            segments.append([token, 0, owner, absolute])
        segments[-1][1] += 1
        segments[-1][3] += 1
        cursor[owner] = absolute + 1
        stage_cursor[owner] += 1
    assert len(segments) <= seq_info.shape[0]
    seq_info[: len(segments)] = torch.tensor(segments, dtype=torch.int32, device="cuda")
    scale = dim**-0.5
    expected = sdpa_varlen(
        q,
        legacy_k,
        legacy_v,
        seq_info,
        scale,
        window=window,
        ring=rows if window else 0,
    )
    fn = _private_batched_sequence_step_fn(
        scale, rows, columns, window, bool(window), None
    )
    args = (
        q,
        k_step,
        v_step,
        position,
        private_k.reshape(1, -1, heads, dim),
        private_v.reshape(1, -1, heads, dim),
        seq_cols,
        seq_info,
        layouts,
        steps,
        permutation,
        k_stage,
        v_stage,
    )
    return fn, args, expected, payload, spans, ends, base + 1


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrivateVarlenTest(unittest.TestCase):
    def check_case(self, owners, previous, window, **kwargs):
        fn, args, expected, payload, spans, ends, allocated = _case(
            owners, previous, window, **kwargs
        )
        actual = fn(*args)
        # Preserve the old key sweep and reduction configuration exactly.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for owner, end in ends.items():
            base, capacity = spans[owner]
            positions = torch.arange(max(0, end - capacity), end, device="cuda")
            physical = positions % capacity if window else positions
            for values, storage in zip(payload[owner], args[4:6]):
                rows = storage.reshape(-1, storage.shape[-2], storage.shape[-1])
                torch.testing.assert_close(
                    rows[base + physical], values[positions], rtol=0, atol=0
                )
                self.assertTrue(torch.isnan(rows[allocated:]).all())

    def test_static_decode_and_prefill(self):
        for window in (0, 31, 257):
            for width in (1, 2, 4, 8, 16, 32, 64, 128, 512):
                with self.subTest(window=window, width=width):
                    self.check_case([0] * width, {0: 1024}, window)

    def test_heterogeneous_interleaved_owners_and_padding(self):
        owners = [0, 1, 0, 2, 1, 2] * 12
        for window in (0, 31):
            with self.subTest(window=window):
                self.check_case(owners, {0: 109, 1: 7, 2: 41}, window, padding=8)

    def test_sixty_four_decode_owners(self):
        for window in (0, 31):
            self.check_case(
                list(range(64)), {owner: owner * 3 + 1 for owner in range(64)}, window
            )

    def test_muse_glimmer_and_gemma_head_geometry(self):
        for dim, heads, groups in ((128, 2, 4), (256, 1, 2)):
            for prefill in (64, 512):
                owners = list(range(8)) + [8] * prefill
                previous = {owner: 1024 for owner in range(9)}
                with self.subTest(dim=dim, prefill=prefill):
                    self.check_case(
                        owners, previous, 127, heads=heads, dim=dim, groups=groups
                    )

    def test_compiled_ring_mutation_order(self):
        fn, args, expected, payload, spans, ends, _ = _case(
            list(range(8)) + [8] * 64, {owner: 1024 for owner in range(9)}, 31
        )
        compiled = torch.compile(fn, fullgraph=True)
        actual = compiled(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for owner, end in ends.items():
            base, capacity = spans[owner]
            positions = torch.arange(end - capacity, end, device="cuda")
            for values, storage in zip(payload[owner], args[4:6]):
                torch.testing.assert_close(
                    storage.reshape(-1, storage.shape[-2], storage.shape[-1])[
                        base + positions % capacity
                    ],
                    values[positions],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
