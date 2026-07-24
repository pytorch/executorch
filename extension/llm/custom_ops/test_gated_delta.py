# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import sys
import unittest

import torch

from executorch.extension.llm.custom_ops import custom_ops  # noqa


class ChannelwiseGatedDeltaRuleTest(unittest.TestCase):
    def _make_inputs(
        self,
        batch_size: int = 2,
        num_heads: int = 3,
        seq_len: int = 4,
        k_head_dim: int = 5,
        v_head_dim: int = 6,
    ):
        query = torch.randn(batch_size, num_heads, seq_len, k_head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, k_head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, v_head_dim)
        # Per-key-channel decay, passed already exponentiated (in (0, 1)).
        decay = torch.rand(batch_size, num_heads, seq_len, k_head_dim)
        beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len))
        initial_state = torch.randn(batch_size, num_heads, k_head_dim, v_head_dim)
        return query, key, value, decay, beta, initial_state

    def _reference_channelwise_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ):
        state = initial_state.clone()
        output = torch.zeros_like(value)

        for token in range(query.size(2)):
            # Per-key-channel decay: [B, H, K, 1], already exponentiated.
            decay_t = decay[:, :, token].unsqueeze(-1)
            beta_t = beta[:, :, token].unsqueeze(-1)
            k_t = key[:, :, token]
            v_t = value[:, :, token]
            q_t = query[:, :, token]

            state = state * decay_t
            v_pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - v_pred) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output[:, :, token] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output, state

    def test_channelwise_gated_delta_rule_matches_reference(self):
        torch.manual_seed(0)

        test_cases = (
            (2, 3, 4, 5, 6),
            (1, 4, 7, 8, 3),
        )

        for case in test_cases:
            with self.subTest(case=case):
                (
                    query,
                    key,
                    value,
                    decay,
                    beta,
                    initial_state,
                ) = self._make_inputs(*case)

                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(
                        query,
                        key,
                        value,
                        decay,
                        beta,
                        initial_state,
                    )
                )

                # Functional op: initial_state must not be mutated.
                initial_state_before = initial_state.clone()
                actual_output, actual_state = (
                    torch.ops.llama.channelwise_gated_delta_rule(
                        query,
                        key,
                        value,
                        decay,
                        beta,
                        initial_state,
                    )
                )

                self.assertTrue(
                    torch.allclose(actual_output, expected_output, atol=1e-5)
                )
                self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-5))
                self.assertTrue(torch.equal(initial_state, initial_state_before))

    def test_channelwise_gated_delta_rule_out_matches_reference(self):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        expected_output, expected_state = self._reference_channelwise_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
        )

        actual_output = torch.empty_like(value)
        actual_final_state = torch.empty_like(initial_state)
        returned_output, returned_state = (
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=actual_final_state,
            )
        )

        self.assertEqual(returned_output.data_ptr(), actual_output.data_ptr())
        self.assertEqual(returned_state.data_ptr(), actual_final_state.data_ptr())
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-5))
        self.assertTrue(torch.allclose(actual_final_state, expected_state, atol=1e-5))

    def test_channelwise_gated_delta_rule_out_invalid_args_raise(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        invalid_key = key[:, :, :, :-1].contiguous()
        actual_output = torch.empty(1)
        actual_final_state = torch.empty(1)

        with self.assertRaises(RuntimeError):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                invalid_key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=actual_final_state,
            )

        self.assertEqual(tuple(actual_output.shape), (1,))
        self.assertEqual(tuple(actual_final_state.shape), (1,))

    def test_channelwise_gated_delta_rule_out_rejects_initial_state_alias(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        actual_output = torch.empty_like(value)

        with self.assertRaisesRegex(
            RuntimeError,
            "final_state_out must not alias initial_state",
        ):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=initial_state,
            )

    def test_channelwise_gated_delta_rule_out_rejects_initial_state_overlap(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs(
            batch_size=1,
            num_heads=1,
            k_head_dim=2,
            v_head_dim=3,
        )
        actual_output = torch.empty_like(value)
        final_state_out = initial_state.as_strided(
            (1, 1, 1, 5),
            (5, 5, 5, 1),
            storage_offset=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "final_state_out must not alias initial_state",
        ):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=final_state_out,
            )

    def test_channelwise_gated_delta_rule_chunked_matches_full_sequence(self):
        torch.manual_seed(0)

        # seq_len > CHUNK_SIZE (32 in the C++ kernel): the single full call runs
        # the internal chunk loop over multiple chunks (6 * 32 + 8), while the
        # per-segment calls are chunked externally as (0, 64), (64, 192),
        # (192, 200) — 2, 4, and 1 partial internal chunks respectively — at
        # different boundaries than the single-call internal loop. Equality
        # validates the inter-chunk state carry independent of outer
        # segmentation.
        # fp32 accumulation over a 32-token chunk needs a looser tol than the
        # single-chunk tests above.
        seq_len = 200
        query, key, value, decay, beta, initial_state = self._make_inputs(
            seq_len=seq_len
        )

        full_output, full_state = torch.ops.llama.channelwise_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
        )

        chunk_state = initial_state
        chunk_outputs = []
        for start, end in ((0, 64), (64, 192), (192, 200)):
            chunk_output, chunk_state = torch.ops.llama.channelwise_gated_delta_rule(
                query[:, :, start:end, :],
                key[:, :, start:end, :],
                value[:, :, start:end, :],
                decay[:, :, start:end, :],
                beta[:, :, start:end],
                chunk_state,
            )
            chunk_outputs.append(chunk_output)

        chunked_output = torch.cat(chunk_outputs, dim=2)
        self.assertTrue(
            torch.allclose(chunked_output, full_output, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(torch.allclose(chunk_state, full_state, atol=1e-3, rtol=1e-3))

    def test_channelwise_gated_delta_rule_multichunk_matches_reference(self):
        torch.manual_seed(0)

        # Drive the prefill route past a single CHUNK_SIZE (32) so the kernel's
        # internal chunk loop runs multiple times against the token-by-token
        # reference: 130 exercises a ragged final chunk (4 * 32 + 2), 256
        # exercises an exact multiple (8 * 32). fp32 accumulation over a full
        # chunk needs a looser tol than the single-chunk tests.
        for seq_len in (130, 256):
            with self.subTest(seq_len=seq_len):
                inputs = self._make_inputs(seq_len=seq_len)
                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(*inputs)
                )

                actual_output, actual_state = (
                    torch.ops.llama.channelwise_gated_delta_rule(*inputs)
                )

                self.assertTrue(
                    torch.allclose(actual_output, expected_output, atol=1e-3, rtol=1e-3)
                )
                self.assertTrue(
                    torch.allclose(actual_state, expected_state, atol=1e-3, rtol=1e-3)
                )

    def test_channelwise_gated_delta_rule_exports(self):
        class Module(torch.nn.Module):
            def forward(self, query, key, value, decay, beta, initial_state):
                return torch.ops.llama.channelwise_gated_delta_rule(
                    query, key, value, decay, beta, initial_state
                )

        inputs = self._make_inputs()

        # Static export: the op must survive as a single graph node (the Meta
        # impl lets it trace without running the real kernel).
        ep = torch.export.export(Module(), inputs)
        targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn("llama.channelwise_gated_delta_rule.default", targets)

        # Dynamic sequence length: one graph shared across prefill/decode.
        seq = torch.export.Dim("seq", min=1, max=128)
        dynamic_shapes = (
            {2: seq},  # query          [B, H, T, K]
            {2: seq},  # key            [B, H, T, K]
            {2: seq},  # value          [B, H, T, V]
            {2: seq},  # decay          [B, H, T, K]
            {2: seq},  # beta           [B, H, T]
            {},  # initial_state  [B, H, K, V] (no T dim)
        )
        ep_dyn = torch.export.export(Module(), inputs, dynamic_shapes=dynamic_shapes)
        self.assertTrue(
            any(
                "channelwise_gated_delta_rule" in str(n.target)
                for n in ep_dyn.graph.nodes
                if n.op == "call_function"
            )
        )

    @unittest.skipUnless(
        sys.platform == "linux",
        "Custom-kernel .pte execution via the Python Runtime is Linux-only: the "
        "channelwise_gated_delta_rule kernel is not registered in the ExecuTorch "
        "pybindings runtime on Windows (custom-op static registration does not "
        "cross the DLL boundary). Eager + export paths are covered on all "
        "platforms by the other tests.",
    )
    def test_channelwise_gated_delta_rule_pte_execution(self):
        # Exports, lowers, and *executes* the op through the ExecuTorch runtime.
        # This is the only test that hits the boxed kernel and its stack
        # arg-count contract: the emitter appends a trailing TensorList to a
        # multi-output out variant, so the runtime passes 9 args, not 8. Eager
        # and export-trace tests never exercise that path.
        import os
        import tempfile

        from executorch.devtools import Inspector
        from executorch.exir import EdgeCompileConfig, to_edge
        from executorch.runtime import Runtime

        op_event_name = "native_call_llama::channelwise_gated_delta_rule.out"

        class Module(torch.nn.Module):
            def forward(self, query, key, value, decay, beta, initial_state):
                return torch.ops.llama.channelwise_gated_delta_rule(
                    query, key, value, decay, beta, initial_state
                )

        runtime = Runtime.get()
        for seq_len in (4, 1):  # T != 1 (chunked route) and T == 1 (decode route)
            with self.subTest(seq_len=seq_len):
                torch.manual_seed(seq_len)
                inputs = self._make_inputs(seq_len=seq_len)
                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(*inputs)
                )

                ep = torch.export.export(Module(), inputs)
                edge = to_edge(
                    ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
                )
                program_buffer = edge.to_executorch().buffer

                # enable_etdump records per-op ProfileEvents. The manually
                # registered boxed kernel only surfaces as native_call_ because
                # its wrapper opens an EventTracerProfileOpScope; this needs the
                # build to define ET_EVENT_TRACER_ENABLED (buckconfig
                # executorch.event_tracer_enabled=true, set on this test target).
                program = runtime.load_program(
                    program_buffer, enable_etdump=True, debug_buffer_size=int(1e7)
                )
                method = program.load_method("forward")
                output, final_state = method.execute(list(inputs))

                self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
                self.assertTrue(torch.allclose(final_state, expected_state, atol=1e-5))

                with tempfile.TemporaryDirectory() as tmp:
                    etdump_path = os.path.join(tmp, "etdump.etdp")
                    debug_path = os.path.join(tmp, "debug.bin")
                    program.write_etdump_result_to_file(etdump_path, debug_path)
                    inspector = Inspector(
                        etdump_path=etdump_path, debug_buffer_path=debug_path
                    )
                    event_names = [
                        event.name
                        for block in inspector.event_blocks
                        for event in block.events
                        if event.name is not None
                    ]

                self.assertIn(
                    op_event_name,
                    event_names,
                    f"{op_event_name!r} missing from ETDump; got "
                    f"{sorted(set(event_names))}",
                )


if __name__ == "__main__":
    unittest.main()
