# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical and export contracts for the DFlash2 draft additions."""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from executorch.examples.models.muse_glimmer.model.dflash2 import (
    CandidateSelector,
    DFlashGroupedConv,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    DFlashConfig,
    DFlashDraftModel,
    DFlashPackedExportModel,
    MuseGlimmerWithDFlash,
)
from executorch.examples.models.muse_glimmer.tests.dflash_reference import (
    DFlash2DraftModel as OfficialDraftModel,
)


class ReferenceTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(96, 32)
        self.lm_head = torch.nn.Linear(32, 96, bias=False)
        self.activation_dtype = torch.float32


class DFlash2OfficialReferenceTest(TestCase):
    def _assert_parity(self, window, pattern, block, contexts, seed):
        torch.manual_seed(seed)
        config = DFlashConfig(
            dim=32,
            n_layers=len(pattern),
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            ffn_dim=48,
            vocab_size=96,
            block_size=block,
            target_layers=[0, 1],
            max_seq_len=128,
            rope_theta=500000.0,
            norm_eps=1e-5,
            mask_token_id=0,
            sliding_window=window,
            sliding_window_pattern=pattern if window else None,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=6,
            selector_top_k=4,
            output_multiplier=0.5,
            final_logit_softcapping=3.0,
        )
        reference_config = SimpleNamespace(
            hidden_size=32,
            num_hidden_layers=len(pattern),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=48,
            vocab_size=96,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attention_bias=False,
            attention_dropout=0.0,
            layer_types=[
                "sliding_attention" if window and sliding else "full_attention"
                for sliding in pattern
            ],
            sliding_window=window,
            is_causal=False,
            num_target_layers=2,
            dflash_config={
                "block_size": block,
                "conv_kernel_size": 2,
                "conv_group_size": 8,
                "selector_rank": 6,
                "selector_top_k": 4,
                "target_layer_ids": [0, 1],
                "mask_token_id": 0,
                "output_multiplier": 0.5,
                "final_logit_softcapping": 3.0,
            },
        )
        reference = OfficialDraftModel(reference_config).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.copy_(1 + 0.1 * torch.randn_like(parameter))
                else:
                    parameter.copy_(0.3 * torch.randn_like(parameter))
        draft = DFlashDraftModel(config, max_context_length=128).eval()
        missing, unexpected = draft.load_state_dict(
            reference.state_dict(), strict=False
        )
        self.assertFalse(unexpected)
        self.assertTrue(all("cache" in name for name in missing), missing)
        target = ReferenceTarget().eval()
        with torch.no_grad():
            target.embed_tokens.weight.normal_()
            target.lm_head.weight.mul_(4)
        wrapper = MuseGlimmerWithDFlash(target, draft, None, config).eval()
        hidden_outputs = []
        hook = draft.register_forward_hook(
            lambda module, inputs, output: hidden_outputs.append(output)
        )
        self.addCleanup(hook.remove)
        context_history = torch.empty(1, 0, 64)
        position = 0
        with torch.no_grad():
            for length in contexts:
                context = torch.randn(1, length, 64)
                context_history = torch.cat((context_history, context), dim=1)
                tokens = torch.zeros(1, block, dtype=torch.long)
                tokens[0, 0] = torch.randint(1, 96, ())
                expected_hidden = reference(
                    position_ids=torch.arange(position + length + block)[None],
                    noise_embedding=target.embed_tokens(tokens),
                    target_hidden=context_history,
                )
                candidates, scores = wrapper.draft_forward(
                    tokens, context, torch.tensor([position])
                )
                torch.testing.assert_close(
                    hidden_outputs.pop(),
                    expected_hidden,
                    atol=1e-4,
                    rtol=0,
                    msg="hidden states differ from the official reference",
                )
                hidden = expected_hidden[:, 1:]
                path, reference_candidates, _ = reference.propose(
                    hidden, tokens[:, 0], target.lm_head, 0.0
                )
                torch.testing.assert_close(
                    candidates.sort(dim=-1).values,
                    reference_candidates.sort(dim=-1).values,
                )
                logits = reference.compute_logits(hidden, target.lm_head)
                selector = reference.candidate_selector
                projected = selector.hidden_projection(hidden)
                expected_scores = torch.empty_like(scores)
                actual_path = []
                previous_index = 0
                for step in range(block - 1):
                    for predecessor_index in range(4):
                        predecessor = (
                            tokens[0, 0]
                            if step == 0
                            else candidates[0, step - 1, predecessor_index]
                        )
                        for current_index in range(4):
                            current = candidates[0, step, current_index]
                            pair = (
                                selector.predecessor_codebook.weight[predecessor]
                                * projected[0, step]
                                * selector.successor_codebook.weight[current]
                            )
                            expected_scores[
                                0, step, predecessor_index, current_index
                            ] = (logits[0, step, current] + pair.sum())
                    previous_index = int(scores[0, step, previous_index].argmax())
                    actual_path.append(int(candidates[0, step, previous_index]))
                torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=0)
                self.assertEqual(actual_path, path[0].tolist())
                position += length

    def test_cached_draft_matches_stateless_official_reference(self):
        contexts = [3, 1, 4, 2, 5, 1, 3, 6, 2]
        for window, pattern, block, lengths, seed in (
            (5, [True, False], 4, contexts, 1),
            (3, [True, True], 4, contexts, 3),
            (None, [False, False], 4, contexts, 5),
            (16, [True, True], 16, [7, 9, 3, 12, 1, 5], 6),
        ):
            with self.subTest(window=window, pattern=pattern, block=block):
                self._assert_parity(window, pattern, block, lengths, seed)

    def test_reference_detects_uniform_window_mask(self):
        def old_uniform_mask(total_context, block, capacity, window):
            slots = torch.arange(capacity, dtype=torch.long)
            positions = slots + ((total_context - 1 - slots) // capacity) * capacity
            visible = (positions >= 0) & (positions >= total_context - window)
            return torch.cat((visible, torch.ones(block, dtype=torch.bool)))[
                None, None, None
            ]

        for window, pattern, block, contexts, seed in (
            (5, [True, False], 4, [3, 1, 4, 2, 5, 1, 3, 6, 2], 1),
            (16, [True, True], 16, [7, 9, 3, 12, 1, 5], 6),
        ):
            with (
                self.subTest(window=window, block=block),
                patch(
                    "executorch.examples.models.muse_glimmer.model.dflash_model.build_dflash_swa_mask",
                    old_uniform_mask,
                ),
                self.assertRaisesRegex(AssertionError, "hidden states"),
            ):
                self._assert_parity(window, pattern, block, contexts, seed)


class DFlash2OperationsTest(TestCase):
    def test_invalid_metadata(self):
        for metadata in (
            {"conv_kernel_size": 2},
            {"selector_top_k": 16},
            {"output_multiplier": float("nan")},
            {"final_logit_softcapping": float("nan")},
            {"final_logit_softcapping": float("inf")},
            {"final_logit_softcapping": float("-inf")},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                DFlashConfig(**metadata)

    def test_disabled_softcap_metadata(self):
        for value in (None, 0, -1, 20):
            with self.subTest(value=value):
                config = DFlashConfig.from_gguf_metadata(
                    {"dflash.final_logit_softcapping": value}
                )
                self.assertEqual(
                    config.final_logit_softcapping, 20 if value == 20 else None
                )

    def test_embedding_scale_matches_packed_draft(self):
        class EchoDraft(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.candidate_selector = None

            def forward(self, embedding, *unused):
                return embedding

        torch.manual_seed(12)
        target = ReferenceTarget().eval()
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        hidden, positions = torch.empty(1, 0, 32), torch.arange(8)
        for scale in (1.0, 2.0, 0.25):
            config = DFlashConfig.from_gguf_metadata(
                {"dflash.embedding_scale": scale, "dflash.block_size": 4}
            )
            draft = EchoDraft(config)
            wrapper = MuseGlimmerWithDFlash(target, draft, None, config)
            expected = target.lm_head(target.embed_tokens(tokens) * scale)
            with self.subTest(scale=scale):
                torch.testing.assert_close(
                    wrapper.draft_forward(tokens, hidden, positions), expected
                )
                for packed in (
                    wrapper.draft_forward_packed,
                    DFlashPackedExportModel(target, draft),
                ):
                    torch.testing.assert_close(
                        packed(tokens, hidden, positions, positions),
                        expected.reshape(1, 8, -1),
                    )

    def test_fixed_block_exports_dynamic_proposal_prefix(self):
        from executorch.examples.models.muse_glimmer.export.common import (
            BoundMethodForward,
        )
        from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
            dflash_mlx_source_transformations,
        )
        from executorch.extension.llm.cache.reference_cache import AttendSpec, MaskKind
        from executorch.extension.llm.cache.update_and_attend import REGISTRY

        for block in (2, 16):
            torch.manual_seed(17)
            config = DFlashConfig(
                dim=32,
                n_layers=1,
                n_heads=4,
                n_kv_heads=2,
                head_dim=8,
                ffn_dim=48,
                vocab_size=96,
                block_size=block,
                target_layers=[0],
                max_seq_len=64,
                conv_kernel_size=2,
                conv_group_size=8,
                selector_rank=6,
                selector_top_k=4,
            )
            draft = DFlashDraftModel(config, max_context_length=64).eval()
            dflash_mlx_source_transformations(draft, use_offgraph_kv_cache=True)
            wrapper = MuseGlimmerWithDFlash(
                ReferenceTarget().eval(), draft, None, config
            )
            inputs = (
                torch.randint(0, 96, (2, block)),
                torch.randn(1, 2, 32),
                torch.arange(2 + 2 * block),
                torch.arange(2 + 2 * block),
            )
            with BoundMethodForward(wrapper, wrapper.draft_forward), torch.no_grad():
                ep = torch.export.export(
                    wrapper,
                    (*inputs, torch.arange(1, block)),
                    dynamic_shapes=(
                        {},
                        {},
                        {},
                        {},
                        (
                            {0: torch.export.Dim("proposals", min=1, max=block - 1)}
                            if block > 2
                            else None
                        ),
                    ),
                    strict=True,
                )
            self.assertEqual(len(ep.graph_signature.user_inputs), 5)
            with patch.object(REGISTRY, "current") as cache, torch.no_grad():
                cache.return_value.update_and_fetch.side_effect = (
                    lambda layer, k, v, pos, queries=2 * block: [
                        AttendSpec(k, v, MaskKind.NONE, queries)
                    ]
                )
                expected_ids, expected_scores = wrapper.draft_forward(
                    *inputs, torch.arange(1, block)
                )
                for count in range(1, block):
                    with self.subTest(block=block, proposals=count):
                        ids, scores = ep.module()(
                            *inputs, torch.full((count,), -7, dtype=torch.long)
                        )
                        torch.testing.assert_close(ids, expected_ids[:, :count])
                        torch.testing.assert_close(scores, expected_scores[:, :count])

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
    def test_dflash2_requires_explicit_attention_mode(self):
        metadata = {
            "dflash.conv_kernel_size": 2,
            "dflash.conv_group_size": 16,
            "dflash.selector_rank": 256,
            "dflash.selector_top_k": 16,
        }
        with self.assertRaisesRegex(ValueError, "explicit dflash.attention.causal"):
            DFlashConfig.from_gguf_metadata(metadata)
        self.assertEqual(
            DFlashConfig.from_gguf_metadata(
                metadata | {"dflash.attention.causal": False}
            ).selector_top_k,
            16,
        )
        self.assertEqual(DFlashConfig.from_gguf_metadata({}).selector_top_k, 0)

    def test_cuda_packed_rejects_dflash2_before_transform(self):
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            prepare_cuda_packed_draft,
        )

        draft = SimpleNamespace(
            config=SimpleNamespace(selector_top_k=4, conv_kernel_size=2)
        )
        with self.assertRaisesRegex(ValueError, "DFlash2 currently requires"):
            prepare_cuda_packed_draft(
                None, None, draft, max_step_tokens=16, max_cells=128, max_owners=2
            )

    def test_invalid_gguf_rejects_before_target_weights(self):
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            export_dflash,
        )
        from gguf import GGUFWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.gguf"
            writer = GGUFWriter(str(path), "dflash")
            writer.add_uint32("dflash.conv_kernel_size", 2)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with patch(
                "executorch.examples.models.muse_glimmer.loaders.checkpoint_loader.load_gguf_model",
                side_effect=AssertionError("target weights accessed"),
            ), self.assertRaisesRegex(ValueError, "convolution and selector metadata"):
                export_dflash(
                    directory,
                    target_gguf="unused.gguf",
                    draft_gguf=str(path),
                    backend="mlx",
                    use_offgraph_kv_cache=True,
                )

    def test_gguf_rejects_causal_attention(self):
        with self.assertRaisesRegex(
            ValueError, "Causal DFlash attention is unsupported"
        ):
            DFlashConfig.from_gguf_metadata({"dflash.attention.causal": True})

    def test_unsupported_export_rejects_before_loading_weights(self):
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            export_dflash,
        )
        from gguf import GGUFWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.gguf"
            writer = GGUFWriter(str(path), "dflash")
            for key, value in {
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 256,
                "selector_top_k": 16,
            }.items():
                writer.add_uint32("dflash." + key, value)
            writer.add_bool("dflash.attention.causal", False)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            for backend, offgraph in (("mlx", False), ("cuda", False), ("cuda", True)):
                with (
                    self.subTest(backend=backend, offgraph=offgraph),
                    patch(
                        "executorch.examples.models.muse_glimmer.loaders."
                        "checkpoint_loader.load_gguf_model",
                        side_effect=AssertionError("target weights accessed"),
                    ),
                    patch(
                        "executorch.extension.llm.export.load.load_checkpoint",
                        side_effect=AssertionError("draft weights accessed"),
                    ),
                    self.assertRaisesRegex(ValueError, "DFlash2 currently requires"),
                ):
                    export_dflash(
                        directory,
                        target_gguf="unused-target.gguf",
                        draft_gguf=str(path),
                        backend=backend,
                        use_offgraph_kv_cache=offgraph,
                    )

    def test_mlx_dflash2_checkpoint_requires_gguf(self):
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            export_dflash,
        )
        from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
            load_dflash_mlx,
        )

        config = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 40,
            "block_size": 4,
            "target_layer_ids": [0],
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "mask_token_id": 39,
        }
        for metadata in (
            {"architectures": ["DFlash2DraftModel"]},
            {"selector_top_k": 3},
            {"dflash_config": {"conv_kernel_size": 2, "selector_rank": 4}},
        ):
            with (
                self.subTest(metadata=metadata),
                tempfile.TemporaryDirectory() as directory,
                patch(
                    "executorch.extension.llm.export.load.load_checkpoint",
                    side_effect=AssertionError("draft weights accessed"),
                ),
            ):
                (Path(directory) / "config.json").write_text(
                    json.dumps(config | metadata)
                )
                with self.assertRaisesRegex(
                    ValueError, "DFlash2 requires --draft-gguf"
                ):
                    load_dflash_mlx(directory, max_seq_len=16)
                with patch(
                    "executorch.examples.models.muse_glimmer.loaders.checkpoint_loader.load_gguf_model",
                    side_effect=AssertionError("target weights accessed"),
                ), self.assertRaisesRegex(ValueError, "DFlash2 requires --draft-gguf"):
                    export_dflash(
                        directory,
                        target_gguf="unused.gguf",
                        draft_mlx=directory,
                        backend="mlx",
                        use_offgraph_kv_cache=True,
                    )

    def test_gguf_preserves_converted_taps_and_selector_metadata(self):
        from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
            dflash_gguf_to_model_key,
        )

        config = DFlashConfig.from_gguf_metadata(
            {
                "dflash.target_layers": [2, 14, 26, 38, 50],
                "dflash.attention.causal": False,
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
            with (
                self.subTest(name=name, shape=shape),
                tempfile.TemporaryDirectory() as directory,
            ):
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
                writer.add_bool("dflash.attention.causal", False)
                writer.add_tensor(name, torch.zeros(shape).numpy())
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                with self.assertRaisesRegex(error, message):
                    load_dflash_gguf(str(path), max_seq_len=64)


class DFlash2MLXTest(TestCase):
    def test_packed_export_isolates_sessions(self):
        from executorch.examples.models.muse_glimmer.tests.test_mlx_pipeline import (
            _require_mlx,
        )

        _require_mlx(self)
        from executorch import exir
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            _export_dflash_mlx,
        )
        from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
            dflash_mlx_source_transformations,
        )
        from executorch.examples.models.muse_glimmer.tests.test_mlx_offgraph import (
            _model,
            CONFIG,
        )
        from executorch.extension.llm.cache.reference_cache import AttendSpec, MaskKind
        from executorch.extension.llm.cache.update_and_attend import REGISTRY

        cfg = DFlashConfig(
            dim=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            head_dim=16,
            ffn_dim=64,
            vocab_size=CONFIG.vocab_size,
            target_layers=[1],
            block_size=4,
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=3,
            max_seq_len=CONFIG.max_seq_len,
            mask_token_id=0,
        )
        torch.manual_seed(73)
        target = _model()
        draft = DFlashDraftModel(cfg).eval()
        reference = MuseGlimmerWithDFlash(
            copy.deepcopy(target), copy.deepcopy(draft), CONFIG, cfg
        ).eval()
        dflash_mlx_source_transformations(
            reference.draft,
            max_context_length=CONFIG.max_seq_len,
            dtype=torch.float16,
            use_offgraph_kv_cache=True,
        )
        reference.to(torch.float16)
        reference.target.activation_dtype = torch.float16
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                exir,
                "to_edge_transform_and_lower",
                wraps=exir.to_edge_transform_and_lower,
            ) as lower,
        ):
            _export_dflash_mlx(
                target,
                CONFIG,
                draft,
                cfg,
                None,
                None,
                directory,
                CONFIG.max_seq_len,
                torch.float16,
                4,
                use_offgraph_kv_cache=True,
            )
        exported = lower.call_args.args[0]["draft_forward"].module()

        def attend(counts):
            def update(layer, k, v, positions):
                specs = []
                offset = 0
                for count in counts:
                    end = offset + count + cfg.block_size
                    specs.append(
                        AttendSpec(
                            k[:, :, offset:end],
                            v[:, :, offset:end],
                            MaskKind.NONE,
                            cfg.block_size,
                        )
                    )
                    offset = end
                return specs

            return update

        with patch.object(REGISTRY, "current") as cache, torch.no_grad():
            for counts in ((0,), (1,), (2, 3), (0, 2, 1)):
                with self.subTest(counts=counts):
                    tokens = torch.randint(
                        0, cfg.vocab_size, (len(counts), cfg.block_size)
                    )
                    hidden = torch.randn(1, sum(counts), cfg.dim, dtype=torch.float16)
                    ctx_positions = torch.cat([torch.arange(count) for count in counts])
                    block_positions = torch.cat(
                        [
                            torch.arange(count, count + cfg.block_size)
                            for count in counts
                        ]
                    )
                    order, offset = [], 0
                    for owner, count in enumerate(counts):
                        order.extend(range(offset, offset + count))
                        order.extend(
                            range(
                                sum(counts) + owner * cfg.block_size,
                                sum(counts) + (owner + 1) * cfg.block_size,
                            )
                        )
                        offset += count
                    cache.return_value.update_and_fetch.side_effect = attend(counts)
                    actual = exported(
                        tokens,
                        hidden,
                        torch.cat((ctx_positions, block_positions)),
                        torch.tensor(order),
                        torch.arange(1, cfg.block_size),
                    )
                    ids, scores, offset = [], [], 0
                    for owner, count in enumerate(counts):
                        cache.return_value.update_and_fetch.side_effect = attend(
                            (count,)
                        )
                        expected = reference.draft_forward(
                            tokens[owner : owner + 1],
                            hidden[:, offset : offset + count],
                            torch.arange(count + cfg.block_size),
                            torch.arange(count + cfg.block_size),
                        )
                        ids.append(expected[0])
                        scores.append(expected[1])
                        offset += count
                    torch.testing.assert_close(actual[0], torch.cat(ids))
                    torch.testing.assert_close(
                        actual[1], torch.cat(scores), atol=0.01, rtol=0.01
                    )
                    for proposals in (1, 2):
                        cache.return_value.update_and_fetch.side_effect = attend(counts)
                        limited = exported(
                            tokens,
                            hidden,
                            torch.cat((ctx_positions, block_positions)),
                            torch.tensor(order),
                            torch.arange(1, proposals + 1),
                        )
                        torch.testing.assert_close(limited[0], actual[0][:, :proposals])
                        torch.testing.assert_close(
                            limited[1], actual[1][:, :proposals], atol=0.01, rtol=0.01
                        )
            with self.assertRaises(RuntimeError):
                exported(
                    torch.zeros(1, 2, dtype=torch.long),
                    torch.zeros(1, 0, cfg.dim),
                    torch.arange(2),
                    torch.arange(2),
                    torch.arange(1, 2),
                )
