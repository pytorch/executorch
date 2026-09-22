# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from unittest import skipUnless, TestCase
from unittest.mock import patch

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

    def test_wide_verification_requires_cuda_and_supported_length(self) -> None:
        for length in (4, 8, 16):
            export_dflash.validate_dflash_export_options("cuda", length)
        for backend, length in (("mlx", 8), ("cuda", 3), ("cuda", 17)):
            with self.subTest(backend=backend, length=length):
                with self.assertRaises(ValueError):
                    export_dflash.validate_dflash_export_options(backend, length)

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_quantized_draft_export_preserves_packed_block_and_prefill_dispatch(self):
        from executorch.backends.cuda.coalesced_int4_tensor import (
            CudaCoalescedInt4Tensor,
        )
        from executorch.examples.models.muse_glimmer.model.dflash_model import (
            DFlashConfig,
            DFlashDraftModel,
        )
        from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            TINY_CONFIG,
        )
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor
        from executorch.extension.llm.export.quant.quantize import quantize_weight
        from executorch.extension.llm.export.quant.recipe import QuantConfig

        for width in (4, 8, 16):
            with self.subTest(width=width):
                target_config = replace(TINY_CONFIG, n_layers=1)
                target = MuseGlimmerModel(target_config).eval().to(torch.bfloat16)
                target.activation_dtype = torch.bfloat16
                draft_config = DFlashConfig(
                    dim=256,
                    n_layers=1,
                    n_heads=4,
                    n_kv_heads=2,
                    head_dim=64,
                    ffn_dim=512,
                    vocab_size=256,
                    block_size=16,
                    target_layers=[0],
                    max_seq_len=64,
                    conv_kernel_size=2,
                    conv_group_size=16,
                    selector_rank=16,
                    selector_top_k=16,
                )
                draft = DFlashDraftModel(draft_config, 64).eval().to(torch.bfloat16)
                for linear in (draft.fc, draft.layers[0].mlp.gate_proj, target.lm_head):
                    quantized = quantize_weight(
                        linear.weight.detach(),
                        QuantConfig(
                            bits=4, group_size=32, symmetric=False, method="min_max"
                        ),
                    )
                    weight = ExportableInt4Tensor.from_int4_tensor(quantized)
                    if linear is target.lm_head:
                        weight = CudaCoalescedInt4Tensor.from_exportable_int4_tensor(
                            weight
                        )
                    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
                with patch(
                    "executorch.exir.to_edge_transform_and_lower"
                ) as lower, patch.object(export_dflash.common, "save_pte"):
                    export_dflash._export_dflash_cuda(
                        target,
                        target_config,
                        draft,
                        draft_config,
                        None,
                        None,
                        "unused",
                        64,
                        torch.bfloat16,
                        0,
                        verification_length=width,
                    )
                methods = lower.call_args.args[0]
                packed_rows = {
                    name: [
                        node.args[0].meta["val"].shape[0]
                        for node in ep.graph.nodes
                        if node.target
                        == torch.ops.executorch_cuda.int4_plain_mm.default
                    ]
                    for name, ep in methods.items()
                }
                self.assertCountEqual(
                    packed_rows["draft_forward"], [width, 16, width - 1]
                )
                self.assertEqual(packed_rows["draft_prefill"], [])
                self.assertEqual(packed_rows["target_prefill_from_embeddings"], [1])
