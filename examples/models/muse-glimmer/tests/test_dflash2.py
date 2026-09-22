# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical and export contracts for the DFlash2 draft additions."""

import copy
import tempfile
from pathlib import Path
from unittest import skipUnless, TestCase

import torch
from executorch.examples.models.muse_glimmer.model.dflash2 import (
    candidate_topk,
    CandidateSelector,
    DFlashGroupedConv,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    DFlashConfig,
    DFlashDraftModel,
    MuseGlimmerWithDFlash,
)


class DFlash2OperationsTest(TestCase):
    def test_three_proposals_preserve_the_full_backbone_block(self):
        torch.manual_seed(17)
        config = DFlashConfig(
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            ffn_dim=64,
            vocab_size=40,
            block_size=16,
            target_layers=[0],
            max_seq_len=64,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=3,
        )
        target = torch.nn.Module()
        target.embed_tokens = torch.nn.Embedding(40, 32)
        target.lm_head = torch.nn.Linear(32, 40, bias=False)
        target.activation_dtype = torch.float32
        draft = DFlashDraftModel(config, max_context_length=64).eval()
        full = MuseGlimmerWithDFlash(target, draft, None, config)
        limited = MuseGlimmerWithDFlash(
            copy.deepcopy(target),
            copy.deepcopy(draft),
            None,
            config,
            max_draft_tokens=3,
        )
        backbone_rows, projected_rows = [], []
        limited.draft.register_forward_pre_hook(
            lambda module, inputs: backbone_rows.append(inputs[0].shape[1])
        )
        limited.target.lm_head.register_forward_pre_hook(
            lambda module, inputs: projected_rows.append(inputs[0].shape[1])
        )
        inputs = (
            torch.randint(0, 40, (1, 16)),
            torch.randn(1, 4, 32),
            torch.tensor([0]),
        )
        with torch.no_grad():
            expected_ids, expected_scores = full.draft_forward(*inputs)
            ids, scores = limited.draft_forward(*inputs)
        self.assertEqual(backbone_rows, [16])
        self.assertEqual(projected_rows, [3])
        torch.testing.assert_close(ids, expected_ids[:, :3])
        torch.testing.assert_close(scores, expected_scores[:, :3])

    def test_large_vocabulary_candidates_match_global_topk(self):
        torch.manual_seed(25)
        for vocab_size in (4096, 4097, 202048):
            logits = torch.randn(1, 3, vocab_size)
            values, ids = candidate_topk(logits, 16)
            expected = torch.topk(logits, 16, dim=-1)
            torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
            torch.testing.assert_close(logits.gather(-1, ids), values, rtol=0, atol=0)
            self.assertTrue(bool(torch.all(ids < vocab_size)))

    def test_convolution_does_not_cross_request_boundaries(self):
        torch.manual_seed(4)
        conv = DFlashGroupedConv(12, taps=3, group_size=4)
        x = torch.randn(2, 5, 12)
        delta = conv.kernel_projection(x).view(2, 5, 2, 3, 3)
        for side in (0, 1):
            expected = torch.zeros_like(x)
            for batch in range(2):
                for position in range(5):
                    for tap in range(min(3, position + 1)):
                        coefficient = conv.base_kernel[side, tap] + delta[
                            batch, position, side, tap
                        ].repeat_interleave(4)
                        expected[batch, position] += (
                            coefficient * x[batch, position - tap]
                        )
            actual = conv._convolve(x, delta[:, :, side], side)
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(conv.prepare(x)[0][1], conv.prepare(x[1:])[0][0])

    def test_convolution_exports_dynamic_block_length(self):
        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = DFlashGroupedConv(16, taps=2, group_size=4)

            def forward(self, x):
                h, coefficients = self.conv.prepare(x)
                return self.conv.finish(h.square(), coefficients)

        model = Wrapped().eval()
        ep = torch.export.export(
            model,
            (torch.randn(2, 8, 16),),
            dynamic_shapes=({1: torch.export.Dim("block", min=2, max=16)},),
        )
        for length in (2, 5, 16):
            x = torch.randn(2, length, 16)
            torch.testing.assert_close(ep.module()(x), model(x))

    def test_selector_scores_condition_on_each_predecessor(self):
        torch.manual_seed(7)
        selector = CandidateSelector(8, 23, 4, 3)
        ids = torch.tensor([[[2, 5, 8], [4, 9, 11], [3, 7, 13]]])
        anchor = torch.tensor([17])
        hidden = torch.randn(1, 3, 8)
        unary = torch.randn(1, 3, 3)
        actual = selector(ids, unary, hidden, anchor)
        projected = selector.hidden_projection(hidden)
        expected = torch.empty_like(actual)
        for step in range(3):
            for previous in range(3):
                pred_id = anchor[0] if step == 0 else ids[0, step - 1, previous]
                for current in range(3):
                    pair = (
                        selector.predecessor_codebook.weight[pred_id]
                        * projected[0, step]
                    )
                    pair = (
                        pair * selector.successor_codebook.weight[ids[0, step, current]]
                    )
                    expected[0, step, previous, current] = (
                        unary[0, step, current] + pair.sum()
                    )
        torch.testing.assert_close(actual, expected)
        ep = torch.export.export(selector, (ids, unary, hidden, anchor))
        torch.testing.assert_close(ep.module()(ids, unary, hidden, anchor), actual)

    def test_legacy_draft_keeps_its_state_dict_contract(self):
        config = DFlashConfig(
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            ffn_dim=64,
            vocab_size=40,
            block_size=4,
            target_layers=[0],
            max_seq_len=16,
        )
        model = DFlashDraftModel(config, max_context_length=16)
        self.assertIsNone(model.candidate_selector)
        self.assertFalse(
            any("conv" in name or "selector" in name for name in model.state_dict())
        )


class DFlash2CheckpointTest(TestCase):
    def test_gguf_preserves_converted_taps_and_selector_metadata(self):
        from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
            dflash_gguf_to_model_key,
        )

        config = DFlashConfig.from_gguf_metadata(
            {
                "dflash.target_layers": [2, 14, 26, 38, 50],
                "dflash.conv_kernel_size": 2,
                "dflash.conv_group_size": 16,
                "dflash.selector_rank": 256,
                "dflash.selector_top_k": 16,
                "dflash.logit_scale": 0.196116135,
                "dflash.final_logit_softcapping": 20,
            }
        )
        self.assertEqual(config.target_layers, [2, 14, 26, 38, 50])
        self.assertEqual(config.selector_top_k, 16)
        self.assertEqual(config.conv_kernel_size, 2)
        self.assertEqual(config.output_multiplier, 0.196116135)
        self.assertEqual(
            dflash_gguf_to_model_key("blk.3.attn_conv_base"),
            "layers.3.attention_conv.base_kernel",
        )
        self.assertEqual(
            dflash_gguf_to_model_key("selector_predecessor.weight"),
            "candidate_selector.predecessor_codebook.weight",
        )

    def test_gguf_rejects_missing_malformed_or_unknown_weights(self):
        from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
            load_dflash_gguf,
        )
        from gguf import GGUFWriter

        cases = (
            ("blk.0.attn_conv_base", (2, 2, 32), RuntimeError, "not found"),
            ("blk.0.attn_conv_base", (2, 2, 31), ValueError, "shape"),
            ("unknown.weight", (32, 32), ValueError, "Unrecognized"),
        )
        for name, shape, error, message in cases:
            with self.subTest(
                name=name, shape=shape
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "draft.gguf"
                writer = GGUFWriter(str(path), "dflash")
                for key, value in {
                    "embedding_length": 32,
                    "block_count": 2,
                    "attention.head_count": 2,
                    "attention.head_count_kv": 1,
                    "attention.key_length": 8,
                    "feed_forward_length": 64,
                    "conv_kernel_size": 2,
                    "conv_group_size": 8,
                    "selector_rank": 4,
                    "selector_top_k": 3,
                }.items():
                    writer.add_uint32("dflash." + key, value)
                writer.add_tensor(name, torch.zeros(shape).numpy())
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                with self.assertRaisesRegex(error, message):
                    load_dflash_gguf(str(path), max_seq_len=64)


class DFlash2CudaTest(TestCase):
    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_large_vocabulary_topk_uses_cuda_kernels(self):
        import executorch.backends.cuda.triton.kernels  # noqa: F401
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )
        from executorch.exir import EdgeCompileConfig, to_edge
        from executorch.exir.dialects._ops import ops as exir_ops

        class Candidates(torch.nn.Module):
            def forward(self, logits):
                return candidate_topk(logits, 16)

        logits = torch.randn(1, 3, 202048, device="cuda")
        ep = torch.export.export(Candidates(), (logits,))
        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        ).exported_program()
        graph = ReplaceEdgeOpWithTritonOpPass()(edge.graph_module).graph_module
        self.assertFalse(
            any(
                node.target == exir_ops.edge.aten.topk.default
                for node in graph.graph.nodes
            )
        )
        values, ids = edge.module()(logits)
        torch.testing.assert_close(
            values, torch.topk(logits, 16).values, rtol=0, atol=0
        )
        torch.testing.assert_close(logits.gather(-1, ids), values, rtol=0, atol=0)

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_padded_context_matches_eager_after_ring_wrap(self):
        from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
            dflash_cuda_source_transformations,
            materialize_dflash_runtime_buffers,
        )

        torch.manual_seed(31)
        config = DFlashConfig(
            dim=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            head_dim=32,
            ffn_dim=128,
            vocab_size=40,
            block_size=16,
            target_layers=[1, 2],
            max_seq_len=128,
            sliding_window=32,
            sliding_window_pattern=[True, True],
            conv_kernel_size=2,
            conv_group_size=16,
            selector_rank=8,
            selector_top_k=4,
        )
        eager = DFlashDraftModel(config, max_context_length=128).eval()
        transformed = copy.deepcopy(eager)
        dflash_cuda_source_transformations(transformed)
        materialize_dflash_runtime_buffers(transformed, torch.bfloat16)
        eager = eager.to(device="cuda", dtype=torch.bfloat16)
        transformed = transformed.to(device="cuda", dtype=torch.bfloat16)
        position = 0
        with torch.inference_mode():
            for rows, block in ((4, 16), (1, 2), (3, 5), (4, 8)) * 6:
                noise = torch.randn(1, block, 64, device="cuda", dtype=torch.bfloat16)
                hidden = torch.randn(1, rows, 128, device="cuda", dtype=torch.bfloat16)
                padded = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
                padded[:, :rows] = hidden
                pos = torch.tensor([position], device="cuda")
                actual = transformed(
                    noise, padded, pos, torch.tensor([rows], device="cuda")
                )
                expected = eager(noise, hidden, pos)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                position += rows
