# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small export exemplars must preserve the full selected-logit API."""

import copy
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from executorch.examples.models.muse_glimmer.export import common
from executorch.examples.models.muse_glimmer.export.export_solo import (
    export_cuda_batching,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    LayerPolicy,
    SequenceReferenceCache,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY


class CapturedPrograms(Exception):
    pass


class SequenceExportInputsTest(unittest.TestCase):
    def test_small_logits_example_preserves_token_and_embedding_capacity(self):
        torch.manual_seed(42)
        config = replace(
            TINY_CONFIG,
            dim=32,
            n_layers=2,
            head_dim=8,
            max_seq_len=128,
            global_attn_cfg="[16,0]",
        )
        target = MuseGlimmerModel(config).to(torch.bfloat16).eval()
        target.activation_dtype = torch.bfloat16
        reference = copy.deepcopy(target)
        captured = {}

        def capture(programs, **kwargs):
            captured.update(programs)
            raise CapturedPrograms

        stub = torch.export.export(torch.nn.Identity(), (torch.ones(2, 2),))
        with (
            tempfile.TemporaryDirectory() as out,
            patch("executorch.exir.to_edge_transform_and_lower", side_effect=capture),
            patch.object(common, "export_vision_encoder", return_value=stub),
            self.assertRaises(CapturedPrograms),
        ):
            export_cuda_batching(
                target,
                config,
                out,
                max_step_tokens=32,
                max_cells=128,
                forward_sizes=(1, 2, 4, 8),
                kv_layout="sequence",
                max_kv_seqs=4,
                vision_model=torch.nn.Identity(),
                pos_embed_table=torch.ones(4, 32),
                max_vision_patches=4,
            )

        cache_config = CacheConfig(
            n_layers=config.n_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            capacity=config.max_seq_len,
            layers=tuple(
                (
                    LayerPolicy.ring(config.layer_window_size(i))
                    if config.layer_window_size(i)
                    else LayerPolicy.flat()
                )
                for i in range(config.n_layers)
            ),
        )
        tokens = torch.randint(config.vocab_size, (1, 32))
        positions = torch.arange(32)
        with torch.no_grad():
            expected = reference(tokens, positions)[0]
            embeddings = reference.embed_text(tokens)
        for name in ("forward_others", "forward_from_embeddings"):
            program = captured[name]
            self.assertLessEqual(program.example_inputs[0][2].numel(), 5)
            call = program.module()
            for count in (5, 9, 32):
                with self.subTest(method=name, selected_rows=count):
                    key = f"export-inputs-{name}-{count}"
                    REGISTRY.install(key, SequenceReferenceCache(cache_config))
                    try:
                        with torch.no_grad(), REGISTRY.active(key):
                            actual = call(
                                tokens if name == "forward_others" else embeddings,
                                positions,
                                torch.arange(count),
                            )
                    finally:
                        REGISTRY.uninstall(key)
                    self.assertEqual(tuple(actual.shape), (count, config.vocab_size))
                    torch.testing.assert_close(
                        actual, expected[:count], atol=5e-2, rtol=0
                    )
