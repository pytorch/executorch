# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from executorch.backends.native.custom_ops import gated_delta_rule_op
from executorch.backends.native.serialization import deserialize_graph
from executorch.backends.native.test.utils import (
    _call_function_targets,
    _get_delegate_blob,
    _lower,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def _inputs(tokens: int, heads: int = 4, dim: int = 16, dtype=torch.float32):
    generator = torch.Generator().manual_seed(tokens)
    qkv = [
        torch.randn(1, tokens, heads, dim, generator=generator, dtype=dtype)
        for _ in range(3)
    ]
    g = -torch.rand(1, tokens, heads, generator=generator).to(dtype)
    beta = torch.rand(1, tokens, heads, generator=generator).to(dtype)
    state = torch.randn(1, heads, dim, dim, generator=generator)
    return (*qkv, g, beta, state)


def _transformers(rule, q, k, v, g, beta, state, qk_l2norm):
    """transformers' rule, its [B, H, K, V] state the op's value-major one."""
    out, final = rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=state.transpose(-1, -2),
        output_final_state=True,
        use_qk_l2norm_in_kernel=qk_l2norm,
    )
    return out, final.transpose(-1, -2)


class _Rule(torch.nn.Module):
    def forward(self, q, k, v, g, beta, state):
        return gated_delta_rule_op(q, k, v, g, beta, state, True)


class GatedDeltaRuleTest(unittest.TestCase):
    def test_reference_is_transformers_recurrent_rule(self) -> None:
        # Query and key go in as FP32: transformers before 5.17 runs the
        # l2norm in the input dtype, later releases in FP32 as the op does.
        for tokens, qk_l2norm, dtype in itertools.product(
            (1, 5), (False, True), (torch.float32, torch.float16, torch.bfloat16)
        ):
            with self.subTest(tokens=tokens, qk_l2norm=qk_l2norm, dtype=dtype):
                q, k, *rest = _inputs(tokens, dtype=dtype)
                got = gated_delta_rule_op(q, k, *rest, qk_l2norm)
                out, final = _transformers(
                    torch_recurrent_gated_delta_rule,
                    q.float(),
                    k.float(),
                    *rest,
                    qk_l2norm,
                )
                want = (out.to(dtype), final)
                for a, b in zip(got, want, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reference_matches_transformers_chunked_rule(self) -> None:
        # Part of a 64-token chunk, and one and a part.
        for tokens in (8, 100):
            with self.subTest(tokens=tokens):
                inputs = _inputs(tokens)
                got = gated_delta_rule_op(*inputs, True)
                want = _transformers(torch_chunk_gated_delta_rule, *inputs, True)
                for a, b in zip(got, want, strict=True):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_output_keeps_query_dtype_and_state_is_fp32(self) -> None:
        inputs = _inputs(3, dtype=torch.bfloat16)
        out, final = gated_delta_rule_op(*inputs)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(final.dtype, torch.float32)
        self.assertEqual(out.shape, inputs[2].shape)
        self.assertEqual(final.shape, inputs[5].shape)

    def test_fp64_inputs_compute_in_fp64(self) -> None:
        inputs = tuple(x.double() for x in _inputs(3))
        out, final = gated_delta_rule_op(*inputs, True)
        want = _transformers(torch_recurrent_gated_delta_rule, *inputs, True)
        self.assertEqual((out.dtype, final.dtype), (torch.float64, torch.float64))
        torch.testing.assert_close(out, want[0].double(), rtol=1e-5, atol=1e-5)

    def test_snapshots_are_the_states_after_the_last_tokens(self) -> None:
        q, k, v, g, beta, state = _inputs(5)
        history = torch.stack([state * 0.5, state * 0.25, state * 0.125], dim=1)
        for tokens in (5, 2):
            with self.subTest(tokens=tokens):
                inputs = (q[:, :tokens], k[:, :tokens], v[:, :tokens])
                inputs += (g[:, :tokens], beta[:, :tokens])
                start = torch.cat((state[:, None], history), dim=1)
                out, states = gated_delta_rule_op(*inputs, start, True, 4)
                after = [
                    gated_delta_rule_op(*(x[:, :n] for x in inputs), state, True)[1]
                    for n in range(tokens, 0, -1)
                ]
                want = torch.stack((after + list(start.unbind(1)))[:4], dim=1)
                torch.testing.assert_close(states, want, rtol=0, atol=0)
                torch.testing.assert_close(
                    out, gated_delta_rule_op(*inputs, state, True)[0], rtol=0, atol=0
                )

    def test_no_tokens_keep_the_input_state(self) -> None:
        q, k, v, g, beta, state = _inputs(0)
        history = torch.stack([state, state * 0.5], dim=1)
        for start, snapshots in ((state, 1), (history, 2)):
            with self.subTest(snapshots=snapshots):
                out, final = gated_delta_rule_op(
                    q, k, v, g, beta, start, True, snapshots
                )
                self.assertEqual(out.shape, v.shape)
                torch.testing.assert_close(final, start, rtol=0, atol=0)

    def test_refuses_snapshots_without_their_slots(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, r"state must have shape \[B, S, H, V, K\]"
        ):
            gated_delta_rule_op(*_inputs(2), True, 3)

    def test_refuses_a_state_that_is_not_value_major(self) -> None:
        q, k, v, g, beta, _ = _inputs(2, dim=16)
        v = torch.randn(1, 2, 4, 8)
        with self.assertRaisesRegex(
            RuntimeError, r"state must have shape \[B, H, V, K\]"
        ):
            gated_delta_rule_op(q, k, v, g, beta, torch.zeros(1, 4, 16, 8))

    def test_lowers_whole_to_the_native_delegate(self) -> None:
        graph = deserialize_graph(_get_delegate_blob(_lower(_Rule(), _inputs(4))))
        self.assertIn(
            "torch.ops.native.gated_delta_rule.default", _call_function_targets(graph)
        )
