# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny reference and real lowering regressions for the MLX off-graph ABI."""

import copy
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from executorch.examples.models.muse_glimmer.export import export_solo
from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader
from executorch.examples.models.muse_glimmer.model import model as model_module
from executorch.examples.models.muse_glimmer.model.model import (
    apply_rotary_emb,
    materialize_runtime_buffers,
    MuseGlimmerModel,
)
from executorch.examples.models.muse_glimmer.tests.test_mlx_pipeline import _require_mlx
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG
from executorch.exir.scalar_type import ScalarType
from executorch.extension.llm.cache.reference_cache import (
    BatchedSequenceReferenceCache,
    CacheConfig,
    LayerPolicy,
    SequenceReferenceCache,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY

CONFIG = replace(
    TINY_CONFIG,
    dim=32,
    n_layers=4,
    n_heads=2,
    n_kv_heads=1,
    head_dim=16,
    vocab_size=48,
    global_attn_cfg="[4,4,4,0]",
)


def _model(config=CONFIG):
    torch.manual_seed(42)
    return MuseGlimmerModel(config).eval()


def _cache(config=CONFIG, batched=False):
    cache_config = CacheConfig(
        n_layers=config.n_layers,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        capacity=config.max_seq_len,
        layers=[
            LayerPolicy.ring(window) if window else LayerPolicy.flat()
            for window in (config.layer_window_size(i) for i in range(config.n_layers))
        ],
    )
    return (
        BatchedSequenceReferenceCache(cache_config)
        if batched
        else SequenceReferenceCache(cache_config)
    )


class MLXOffgraphTest(unittest.TestCase):
    def setUp(self):
        self.mlx = _require_mlx(self)
        self.addCleanup(REGISTRY.uninstall, "mg_test")

    def test_gguf_defers_cache_allocation(self):
        config = replace(CONFIG, fuse_qkv=False, fuse_gate_up=False)
        state = _model(config).state_dict()
        with patch.object(
            checkpoint_loader, "_atomic_sd_from_gguf", return_value=state
        ):
            legacy, _ = checkpoint_loader.load_gguf_model(
                "unused.gguf", backend="mlx", config=replace(config)
            )
            self.assertTrue(
                all(
                    layer.self_attn.kv_cache.k_cache.device.type == "cpu"
                    for layer in legacy.layers
                )
            )
            with patch.object(
                model_module,
                "materialize_runtime_buffers",
                side_effect=AssertionError("eager cache allocation"),
            ):
                deferred, _ = checkpoint_loader.load_gguf_model(
                    "unused.gguf",
                    backend="mlx",
                    config=replace(config),
                    defer_runtime_buffers=True,
                )
        self.assertTrue(
            all(
                layer.self_attn.kv_cache.k_cache.device.type == "meta"
                for layer in deferred.layers
            )
        )
        with (
            patch.object(
                self.mlx, "MLXKVCache", side_effect=AssertionError("flat cache")
            ),
            patch.object(
                self.mlx, "MLXRingKVCache", side_effect=AssertionError("ring cache")
            ),
        ):
            self.mlx.mlx_source_transformations(
                deferred, dtype=torch.float16, use_offgraph_kv_cache=True
            )
            materialize_runtime_buffers(deferred, dtype=torch.float16)
        self.assertFalse(
            any("kv_cache" in name for name, _ in deferred.named_buffers())
        )
        self.assertTrue(all(p.device.type == "cpu" for p in deferred.parameters()))

    def test_ring_wrap_parity(self):
        original = _model()
        transformed = copy.deepcopy(original)
        self.mlx.mlx_source_transformations(
            transformed, dtype=torch.float32, use_offgraph_kv_cache=True
        )
        REGISTRY.install("mg_test", _cache())
        # Cross the ring's physical boundary, then continue with single-token decode.
        with REGISTRY.active("mg_test"), torch.no_grad():
            for start, width in ((0, 3), (3, 3), (6, 3), (9, 1)):
                tokens = torch.arange(start, start + width).unsqueeze(0)
                positions = torch.arange(start, start + width)
                embeddings = transformed.mlx_embed_text(tokens)
                expected = original.prefill_from_embeds(embeddings, positions)
                actual = transformed.mlx_prefill_forward(
                    embeddings, positions, torch.arange(width)
                )
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_packed_disjoint_positions(self):
        model = _model()
        self.mlx.mlx_source_transformations(
            model, dtype=torch.float32, use_offgraph_kv_cache=True
        )
        packed = _cache(batched=True)
        independent = [_cache(), _cache()]
        with torch.no_grad():
            for step in (([1, 2, 3], [7]), ([4, 5], [8, 9]), ([6], [10])):
                tokens = torch.tensor([step[0] + step[1]])
                positions = torch.cat(
                    [
                        torch.arange(packed.pos(i), packed.pos(i) + len(part))
                        for i, part in enumerate(step)
                    ]
                )
                selected = torch.tensor([len(step[0]) - 1, tokens.shape[1] - 1])
                packed.declare_step([0] * len(step[0]) + [1] * len(step[1]))
                REGISTRY.install("mg_test", packed)
                with REGISTRY.active("mg_test"):
                    actual = model(tokens, positions, selected)
                expected = []
                for i, part in enumerate(step):
                    cache = independent[i]
                    REGISTRY.install("mg_test", cache)
                    with REGISTRY.active("mg_test"):
                        expected.append(
                            model(
                                torch.tensor([part]),
                                torch.arange(cache.used(0), cache.used(0) + len(part)),
                                torch.tensor([len(part) - 1]),
                            )
                        )
                torch.testing.assert_close(actual, torch.cat(expected, dim=1))

    def test_rope_large_positions_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = _model().to(dtype)
                self.mlx.mlx_source_transformations(
                    model, dtype=dtype, use_offgraph_kv_cache=True
                )
                model.to(dtype)
                attn = model.layers[0].self_attn
                self.assertEqual(attn.inv_freq.dtype, torch.float32)
                x = torch.randn(1, 4, CONFIG.dim, dtype=dtype)
                positions = torch.tensor([10000, 10001, 2, 3])
                # Inspect real projections/RoPE before the cache consumes large positions.
                with patch.object(
                    self.mlx,
                    "update_and_attend",
                    side_effect=lambda q, *args: torch.zeros_like(q),
                ) as attend, torch.no_grad():
                    attn(x, positions)
                    qkv = attn.qkv_proj(attn.qkv_proj_norm(x))
                    q = attn.q_norm(
                        qkv[..., : attn.q_dim].reshape(
                            1, 4, attn.n_heads, attn.head_dim
                        )
                    ).transpose(1, 2)
                    k = attn.k_norm(
                        qkv[..., attn.q_dim : attn.q_dim + attn.kv_dim].reshape(
                            1, 4, attn.n_kv_heads, attn.head_dim
                        )
                    ).transpose(1, 2)
                    frequencies = 1.0 / (
                        CONFIG.rope_theta
                        ** (
                            torch.arange(0, CONFIG.head_dim, 2).float()
                            / CONFIG.head_dim
                        )
                    )
                    phase = positions.float().outer(frequencies)[None, None]
                    expected_q, expected_k = apply_rotary_emb(
                        q, k, phase.cos(), phase.sin()
                    )
                actual_q, actual_k, _, actual_positions, *_ = attend.call_args.args
                torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
                torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
                torch.testing.assert_close(actual_positions, positions[:, None])

    def test_selected_rows_before_head(self):
        model = _model()
        self.mlx.mlx_source_transformations(
            model, dtype=torch.float32, use_offgraph_kv_cache=True
        )
        positions = torch.arange(4)
        rows = torch.tensor([3, 0, 3])
        embeddings = model.mlx_embed_text(torch.tensor([[1, 2, 3, 4]]))
        cache = _cache()
        REGISTRY.install("mg_test", cache)
        with REGISTRY.active("mg_test"), torch.no_grad():
            full = model.mlx_prefill_forward(embeddings, positions, torch.arange(4))
            cache.reset()
            shapes = []
            handle = model.lm_head.register_forward_pre_hook(
                lambda _, args: shapes.append(args[0].shape)
            )
            self.addCleanup(handle.remove)
            selected = model.mlx_prefill_forward(embeddings, positions, rows)
        self.assertEqual(shapes, [torch.Size([1, 3, CONFIG.dim])])
        torch.testing.assert_close(selected, full[:, rows])

    def test_export_bounded_axes_and_decode(self):
        class StopBeforeLowering(Exception):
            pass

        for max_width in (1, 2, 8):
            with self.subTest(max_width=max_width):
                programs = {}

                def capture(exported, programs=programs, **kwargs):
                    programs.update(exported)
                    raise StopBeforeLowering

                with (
                    patch(
                        "executorch.exir.to_edge_transform_and_lower",
                        side_effect=capture,
                    ),
                    self.assertRaises(StopBeforeLowering),
                ):
                    export_solo.export_and_lower(
                        _model(),
                        CONFIG,
                        ".",
                        backend="mlx",
                        activation_dtype=torch.float16,
                        max_prefill_chunk=max_width,
                        use_offgraph_kv_cache=True,
                    )
                embed = programs["embed_text"].module()
                forward = programs["forward_from_embeddings"].module()
                cache = _cache()
                REGISTRY.install("mg_test", cache)
                shapes = (
                    [(1, 1)]
                    if max_width == 1
                    else [(1, 1), (max_width, 1), (1, max_width)]
                )
                with REGISTRY.active("mg_test"), torch.no_grad():
                    for width, rows in shapes:
                        cache.reset()
                        embeddings = embed(torch.zeros((1, width), dtype=torch.long))
                        self.assertEqual(embeddings.shape, (1, width, CONFIG.dim))
                        self.assertEqual(embeddings.dtype, torch.float16)
                        logits = forward(
                            embeddings,
                            torch.arange(width),
                            torch.zeros(rows, dtype=torch.long),
                        )
                        self.assertEqual(logits.shape, (1, rows, CONFIG.vocab_size))
                        self.assertEqual(logits.dtype, torch.float32)
                    with self.assertRaises(RuntimeError):
                        embed(torch.zeros((1, max_width + 1), dtype=torch.long))
                    with self.assertRaises(RuntimeError):
                        forward(
                            torch.zeros((1, 1, CONFIG.dim), dtype=torch.float16),
                            torch.zeros(1, dtype=torch.long),
                            torch.zeros(max_width + 1, dtype=torch.long),
                        )

    def test_legacy_forward_last_logits(self):
        original = _model()
        model = copy.deepcopy(original)
        self.mlx.mlx_source_transformations(model, dtype=torch.float32, max_write_len=4)
        with torch.no_grad():
            for tokens, positions in (([[1, 2, 3]], [0, 1, 2]), ([[4]], [3])):
                inputs = (torch.tensor(tokens), torch.tensor(positions))
                embeddings = model.mlx_embed_text(inputs[0])
                expected = original.prefill_from_embeds(embeddings, inputs[1])[:, -1]
                actual = model(*inputs)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_mlx_lowering_batching_abi(self):
        from executorch.backends.mlx import preprocess
        from executorch.backends.mlx.serialization.mlx_graph_schema import (
            UpdateAndAttendNode,
        )

        try:
            from executorch.runtime import Runtime, Verification
        except ImportError as e:
            self.skipTest(f"ExecuTorch runtime required: {e}")

        for dtype, scalar in (
            (torch.float16, ScalarType.HALF),
            (torch.bfloat16, ScalarType.BFLOAT16),
        ):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                graphs = []
                serialize = preprocess.serialize_mlx_graph

                def capture_graph(graph, graphs=graphs, serialize=serialize):
                    graphs.append(graph)
                    return serialize(graph)

                with patch.object(
                    preprocess, "serialize_mlx_graph", side_effect=capture_graph
                ):
                    export_solo.export_and_lower(
                        _model(),
                        CONFIG,
                        directory,
                        backend="mlx",
                        activation_dtype=dtype,
                        max_prefill_chunk=8,
                        use_offgraph_kv_cache=True,
                    )
                program = Runtime.get().load_program(
                    directory + "/model.pte", verification=Verification.Minimal
                )
                self.assertTrue(
                    {"embed_text", "forward_from_embeddings"}.issubset(
                        program.method_names
                    )
                )
                self.assertNotIn("decode_from_embedding", program.method_names)
                self.assertNotIn("get_mutable_buffer_metadata", program.method_names)
                forward = program.metadata("forward_from_embeddings")
                self.assertEqual(forward.num_inputs(), 3)
                self.assertEqual(forward.num_outputs(), 1)
                self.assertEqual(forward.input_tensor_meta(0).dtype(), int(scalar))
                self.assertEqual(forward.input_tensor_meta(2).sizes(), (8,))
                self.assertEqual(
                    forward.input_tensor_meta(2).dtype(), int(ScalarType.LONG)
                )
                self.assertEqual(
                    forward.output_tensor_meta(0).sizes(), (1, 8, CONFIG.vocab_size)
                )
                self.assertEqual(
                    forward.output_tensor_meta(0).dtype(), int(ScalarType.FLOAT)
                )
                for name, value in {
                    "get_logits_to_keep_mode": 2,
                    "get_activation_dtype": int(scalar),
                    "get_max_seq_len": 8,
                    "get_max_prefill_chunk": 8,
                    "get_max_context_len": CONFIG.max_seq_len,
                    "get_n_caches": CONFIG.n_layers,
                }.items():
                    self.assertEqual(
                        program.load_method(name).execute([]), [value], name
                    )
                for name, values in {
                    "get_windows": [
                        CONFIG.layer_window_size(i) for i in range(CONFIG.n_layers)
                    ],
                    "get_kv_heads": [CONFIG.n_kv_heads] * CONFIG.n_layers,
                    "get_head_dims": [CONFIG.head_dim] * CONFIG.n_layers,
                }.items():
                    actual = program.load_method(name).execute([])[0]
                    torch.testing.assert_close(
                        actual, torch.tensor(values, dtype=torch.int32)
                    )
                # Both methods delegate fully, without graph-owned cache buffers.
                self.assertEqual(len(graphs), 2)
                cache_ops = [
                    instruction.op
                    for graph in graphs
                    for chain in graph.instruction_chains
                    for instruction in chain.instructions
                    if isinstance(instruction.op, UpdateAndAttendNode)
                ]
                self.assertEqual(
                    [op.layer_id for op in cache_ops], list(range(CONFIG.n_layers))
                )
                self.assertTrue(
                    all(graph.num_mutable_buffer_tensors == 0 for graph in graphs)
                )


if __name__ == "__main__":
    # run.py uses runpy without installing this module as sys.modules["__main__"].
    import sys

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MLXOffgraphTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
