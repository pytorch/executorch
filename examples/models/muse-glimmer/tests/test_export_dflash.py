# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest import TestCase

import torch

from executorch.examples.models.muse_glimmer.export import export_dflash


class DFlashExportOptionsTest(TestCase):
    def test_cuda_uses_delegate_mutable_buffer_sharing(self) -> None:
        self.assertFalse(export_dflash._share_graph_mutable_buffers("cuda"))

    def test_mlx_uses_graph_mutable_buffer_sharing(self) -> None:
        self.assertTrue(export_dflash._share_graph_mutable_buffers("mlx"))

    def test_cuda_embed_text_covers_target_prefill(self) -> None:
        self.assertEqual(export_dflash._embed_text_max_len("cuda", 4, 4096), 4096)

    def test_mlx_embed_text_uses_target_bound(self) -> None:
        self.assertEqual(export_dflash._embed_text_max_len("mlx", 2048, 0), 2048)

    def test_draft_prefill_uses_sliding_window_bound(self) -> None:
        config = type(
            "Config",
            (),
            {"sliding_window": 2048, "sliding_window_pattern": [True] * 3},
        )()
        self.assertEqual(export_dflash._max_draft_prefill_len(config, 4096), 2048)

    def test_global_draft_prefill_uses_target_bound(self) -> None:
        config = type(
            "Config",
            (),
            {"sliding_window": 2048, "sliding_window_pattern": [True, False]},
        )()
        self.assertEqual(export_dflash._max_draft_prefill_len(config, 4096), 4096)

    def test_cuda_vision_options_are_valid(self) -> None:
        self.assertIsNone(export_dflash.validate_dflash_export_options("cuda"))

    def test_mlx_vision_options_are_valid(self) -> None:
        self.assertIsNone(export_dflash.validate_dflash_export_options("mlx"))

    def test_callable_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DFlash backend: cpu"):
            export_dflash.validate_dflash_export_options("cpu")

    def test_cuda_speculative_chain_is_device_resident(self) -> None:
        configs = export_dflash._cuda_propagate_device_config()
        for method in (
            "embed_text",
            "target_forward_from_embeddings",
            "target_prefill_from_embeddings",
            "dflash_sample_tokens",
            "dflash_verify_speculative",
        ):
            with self.subTest(method=method):
                self.assertTrue(configs[method].skip_h2d_for_method_inputs)
                self.assertTrue(configs[method].skip_d2h_for_method_outputs)
        for method in ("draft_forward", "draft_prefill"):
            with self.subTest(method=method):
                self.assertFalse(configs[method].skip_h2d_for_method_inputs)
                self.assertTrue(configs[method].skip_d2h_for_method_outputs)

    def test_cuda_sampler_methods_accept_any_proposal_count(self) -> None:
        methods = export_dflash._export_cuda_sampler_methods(
            max_draft_tokens=3, vocab_size=5
        )
        scalars = (torch.tensor([0.0]), torch.tensor([0]), torch.tensor([1.0]))
        for proposals in (1, 3):
            with self.subTest(proposals=proposals):
                tokens, probabilities = methods["dflash_sample_tokens"].module()(
                    torch.randn(proposals, 5), *scalars
                )
                result = methods["dflash_verify_speculative"].module()(
                    torch.randn(proposals + 1, 5),
                    probabilities,
                    torch.cat([torch.tensor([4]), tokens]),
                    *scalars,
                    torch.tensor([False]),
                )
                self.assertEqual(tokens.shape, torch.Size([proposals]))
                self.assertEqual(result.shape, torch.Size([proposals + 3]))
                self.assertEqual(result.dtype, torch.int64)
