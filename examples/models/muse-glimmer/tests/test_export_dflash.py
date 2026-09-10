# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest import TestCase

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

    def test_cuda_device_config_has_exact_method_set(self) -> None:
        configs = export_dflash._cuda_propagate_device_config()
        self.assertEqual(
            set(configs),
            {
                "embed_text",
                "target_forward_from_embeddings",
                "target_prefill_from_embeddings",
                "draft_forward",
                "draft_prefill",
            },
        )

    def test_cuda_target_methods_are_device_in_and_out(self) -> None:
        configs = export_dflash._cuda_propagate_device_config()
        for method in (
            "embed_text",
            "target_forward_from_embeddings",
            "target_prefill_from_embeddings",
        ):
            with self.subTest(method=method):
                self.assertTrue(configs[method].skip_h2d_for_method_inputs)
                self.assertTrue(configs[method].skip_d2h_for_method_outputs)

    def test_cuda_draft_methods_are_host_in_and_device_out(self) -> None:
        configs = export_dflash._cuda_propagate_device_config()
        for method in ("draft_forward", "draft_prefill"):
            with self.subTest(method=method):
                self.assertFalse(configs[method].skip_h2d_for_method_inputs)
                self.assertTrue(configs[method].skip_d2h_for_method_outputs)
