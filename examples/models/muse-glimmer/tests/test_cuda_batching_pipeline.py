# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the batched CUDA export (export_solo_batching.py) on a tiny model.

The eager checks run the off-graph model against the neutral cache's
references; the export check lowers ``decode`` and ``prefill`` and inspects the
artifact. Requires CUDA for the export.

    python -m pytest examples/models/muse-glimmer/tests/test_cuda_batching_pipeline.py -v
"""

import copy
import json
import os
import tempfile
import unittest

import executorch.backends.cuda.quantize_op_dispatch as _quantize_op_dispatch  # noqa: F401
import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    load_prequantized_model,
)
from executorch.examples.models.muse_glimmer.export.export_solo_batching import (
    cell_manifest,
    export_batching,
    MAX_CELLS_METHOD,
    step_forward,
)
from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
    enable_offgraph_kv_cache,
)
from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
    build_random_tiny_model,
    save_checkpoint,
    TINY_CONFIG,
)
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    CellReferenceCache,
    LayerPolicy,
    SequenceReferenceCache,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY


def _cache_config(model, capacity: int) -> CacheConfig:
    attn = model.layers[0].self_attn
    return CacheConfig(
        n_layers=TINY_CONFIG.n_layers,
        n_kv_heads=attn.n_kv_heads,
        head_dim=attn.head_dim,
        capacity=capacity,
        layers=tuple(
            (
                LayerPolicy.ring(layer.self_attn.window_size)
                if layer.self_attn.is_sliding
                else LayerPolicy.flat()
            )
            for layer in model.layers
        ),
    )


class BatchingStepForwardTest(unittest.TestCase):
    """The exported step computes what the model does, per packed sequence."""

    def setUp(self) -> None:
        self.model = build_random_tiny_model()
        self.off_graph = copy.deepcopy(self.model)
        enable_offgraph_kv_cache(self.off_graph, 16)

    def _install(self, cache) -> str:
        key = f"muse-batching-{id(self)}-{id(cache)}"
        REGISTRY.install(key, cache)
        self.addCleanup(REGISTRY.uninstall, key)
        return key

    def test_selected_rows_match_the_full_forward(self) -> None:
        key = self._install(
            SequenceReferenceCache(_cache_config(self.off_graph, TINY_CONFIG.max_seq_len))
        )
        generator = torch.Generator().manual_seed(0)
        start = 0
        for length, keep in ((12, [3, 11]), (1, [0]), (6, [5])):
            tokens = torch.randint(
                0, TINY_CONFIG.vocab_size, (1, length), generator=generator
            )
            input_pos = torch.arange(start, start + length)
            with torch.no_grad():
                expected = self.model(tokens, input_pos)[0, keep]
                with REGISTRY.active(key):
                    actual = step_forward(
                        self.off_graph, tokens, input_pos, torch.tensor(keep)
                    )
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(tuple(actual.shape), (len(keep), TINY_CONFIG.vocab_size))
            self.assertLess((expected - actual).abs().max().item(), 5e-2)
            start += length

    def test_packed_sequences_match_each_alone(self) -> None:
        # Two prompts prefilled in one step, then decoded together, over the
        # neutral cell cache: each sequence's logits must equal running it on
        # its own -- per-token RoPE, windows and caches all kept apart.
        generator = torch.Generator().manual_seed(1)
        a = torch.randint(0, TINY_CONFIG.vocab_size, (1, 20), generator=generator)
        b = torch.randint(0, TINY_CONFIG.vocab_size, (1, 7), generator=generator)
        a_next = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1), generator=generator)
        b_next = torch.randint(0, TINY_CONFIG.vocab_size, (1, 1), generator=generator)

        def alone(prompt, next_token):
            key = self._install(
                SequenceReferenceCache(
                    _cache_config(self.off_graph, TINY_CONFIG.max_seq_len)
                )
            )
            length = prompt.shape[1]
            with torch.no_grad(), REGISTRY.active(key):
                first = step_forward(
                    self.off_graph,
                    prompt,
                    torch.arange(length),
                    torch.tensor([length - 1]),
                )
                second = step_forward(
                    self.off_graph,
                    next_token,
                    torch.tensor([length]),
                    torch.tensor([0]),
                )
            return first[0], second[0]

        a_first, a_second = alone(a, a_next)
        b_first, b_second = alone(b, b_next)

        cells = CellReferenceCache(_cache_config(self.off_graph, 64))
        key = self._install(cells)
        with torch.no_grad(), REGISTRY.active(key):
            cells.declare_step([0] * 20 + [1] * 7)
            prefill = step_forward(
                self.off_graph,
                torch.cat([a, b], dim=1),
                torch.cat([torch.arange(20), torch.arange(7)]),
                torch.tensor([19, 26]),
            )
            cells.declare_step([1, 0])
            decode = step_forward(
                self.off_graph,
                torch.cat([b_next, a_next], dim=1),
                torch.tensor([7, 20]),
                torch.tensor([0, 1]),
            )
        for actual, expected in (
            (prefill[0], a_first),
            (prefill[1], b_first),
            (decode[1], a_second),
            (decode[0], b_second),
        ):
            self.assertLess((expected - actual).abs().max().item(), 5e-2)
            self.assertEqual(int(expected.argmax()), int(actual.argmax()))

    def test_cell_manifest_keeps_every_layer(self) -> None:
        model = build_random_tiny_model()
        manifest = json.loads(cell_manifest(enable_offgraph_kv_cache(model, 8), 128))
        self.assertEqual(manifest["layout"], "cell")
        self.assertEqual(manifest["max_cells"], 128)
        self.assertEqual(manifest["max_write"], 8)
        self.assertEqual(len(manifest["layers"]), TINY_CONFIG.n_layers)


class BatchingExportTest(unittest.TestCase):
    MAX_STEP = 16
    MAX_CELLS = 128

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_exports_decode_and_prefill_over_a_cell_cache(self) -> None:
        from executorch.backends.cuda.passes.lower_offgraph_kv import (
            OFFGRAPH_KV_COMPILE_SPEC,
            OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
        )
        from executorch.exir.lowered_backend_module import get_lowered_submodules

        with (
            tempfile.TemporaryDirectory() as ckpt_dir,
            tempfile.TemporaryDirectory() as out_dir,
        ):
            save_checkpoint(ckpt_dir)
            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            program = export_batching(
                model,
                config,
                out_dir,
                max_step_tokens=self.MAX_STEP,
                max_cells=self.MAX_CELLS,
            )
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))
            self.assertTrue(
                any(name.endswith(".ptd") for name in os.listdir(out_dir))
            )

            methods = set(program.methods)
            self.assertTrue({"decode", "prefill"}.issubset(methods))
            self.assertFalse(
                {"embed_text", "forward_from_embeddings", "decode_from_embedding"}
                & methods
            )
            for name in ("decode", "prefill"):
                graph = program.exported_program(name).graph_module
                lowered = get_lowered_submodules(graph)
                self.assertEqual(len(lowered), 1, name)
                specs = {
                    spec.key: spec.value for spec in lowered[0][1].compile_specs
                }
                manifest = json.loads(specs[OFFGRAPH_KV_COMPILE_SPEC])
                self.assertEqual(manifest["layout"], "cell")
                self.assertEqual(manifest["max_cells"], self.MAX_CELLS)
                self.assertEqual(manifest["max_write"], self.MAX_STEP)
                # The step width must name the delegate input carrying T, which
                # the backend checks is the cache ops' position. The delegate's
                # inputs need not follow the method's order: prefill passes
                # logits_to_keep ahead of input_pos.
                index, dim = map(
                    int, specs[OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC].decode().split(":")
                )
                call = next(
                    n
                    for n in graph.graph.nodes
                    if "executorch_call_delegate" in str(n.target)
                )
                tokens = next(
                    n for n in graph.graph.nodes if n.op == "placeholder"
                    and n.name == "tokens"
                )
                self.assertEqual(
                    str(call.args[1 + index].meta["val"].shape[dim]),
                    str(tokens.meta["val"].shape[1]),
                    name,
                )

            from executorch.runtime import Runtime, Verification

            loaded = Runtime.get().load_program(
                os.path.join(out_dir, "model.pte"), verification=Verification.Minimal
            )

            def constant(name):
                return loaded.load_method(name).execute([])[0]

            self.assertEqual(constant(MAX_CELLS_METHOD), self.MAX_CELLS)
            self.assertEqual(constant("get_max_context_len"), TINY_CONFIG.max_seq_len)
            self.assertEqual(constant("get_n_caches"), TINY_CONFIG.n_layers)
            self.assertEqual(constant("get_max_prefill_chunk"), self.MAX_STEP)
            self.assertEqual(constant("get_min_prefill_chunk"), 5)
            # LogitsToKeepMode::Selected.
            self.assertEqual(constant("get_logits_to_keep_mode"), 2)
