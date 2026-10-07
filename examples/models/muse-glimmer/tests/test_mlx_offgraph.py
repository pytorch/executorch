# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny CPU/reference and real lowering tests for the opt-in MLX solo ABI."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from executorch.examples.models.muse_glimmer.export import common, export_solo
from executorch.examples.models.muse_glimmer.model.model import (
    apply_rotary_emb,
    materialize_runtime_buffers,
    MuseGlimmerModel,
)
from executorch.examples.models.muse_glimmer.source_transformations import mlx
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
        self.addCleanup(REGISTRY.uninstall, "mg_test")

    def test_no_graph_cache_allocation_even_from_meta(self):
        with torch.device("meta"):
            model = MuseGlimmerModel(CONFIG)
        with (
            patch.object(mlx, "MLXKVCache", side_effect=AssertionError("flat cache")),
            patch.object(
                mlx, "MLXRingKVCache", side_effect=AssertionError("ring cache")
            ),
        ):
            mlx.mlx_source_transformations(model, use_offgraph_kv_cache=True)
            materialize_runtime_buffers(model, dtype=torch.float16)
        for i, layer in enumerate(model.layers):
            self.assertEqual(layer.self_attn.layer_idx, i)
            self.assertFalse(hasattr(layer.self_attn, "kv_cache"))
        self.assertFalse(
            any(
                "cache" in n and n != "cache_positions"
                for n, _ in model.named_buffers()
            )
        )

    def test_loader_defers_original_cache_allocation_only_when_requested(self):
        from executorch.examples.models.muse_glimmer.loaders import checkpoint_loader
        from executorch.examples.models.muse_glimmer.model import model as model_module

        config = replace(CONFIG, fuse_qkv=False, fuse_gate_up=False)
        state = _model(config).state_dict()
        with patch.object(
            model_module,
            "materialize_runtime_buffers",
            wraps=materialize_runtime_buffers,
        ) as materialize:
            legacy = checkpoint_loader._finalize(
                state, "mlx", replace(config), torch.float16
            )
            materialize.assert_called_once()
            self.assertTrue(
                all(
                    layer.self_attn.kv_cache.k_cache.device.type == "cpu"
                    for layer in legacy.layers
                )
            )
            materialize.reset_mock()
            deferred = checkpoint_loader._finalize(
                state, "mlx", replace(config), torch.float16, defer_runtime_buffers=True
            )
            materialize.assert_not_called()
            self.assertTrue(
                all(
                    layer.self_attn.kv_cache.k_cache.device.type == "meta"
                    for layer in deferred.layers
                )
            )
        mlx.mlx_source_transformations(
            deferred, dtype=torch.float16, use_offgraph_kv_cache=True
        )
        materialize_runtime_buffers(deferred, dtype=torch.float16)
        self.assertFalse(
            any("kv_cache" in name for name, _ in deferred.named_buffers())
        )
        self.assertTrue(all(p.device.type == "cpu" for p in deferred.parameters()))

    def test_mlx_dispatch_forwards_opt_in(self):
        with patch.object(export_solo, "_export_mlx") as lower:
            export_solo.export_and_lower(
                _model(),
                CONFIG,
                ".",
                backend="mlx",
                max_prefill_chunk=8,
                use_offgraph_kv_cache=True,
            )
            self.assertTrue(lower.call_args.kwargs["use_offgraph_kv_cache"])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            export_solo.export_and_lower(
                _model(),
                CONFIG,
                ".",
                backend="mlx",
                use_offgraph_kv_cache=True,
                use_turboquant=True,
            )

    def test_invalid_offgraph_options_fail_before_export(self):
        cases = (
            ({"backend": "cuda"}, "only supported"),
            ({"activation_dtype": torch.float32}, "float16 or bfloat16"),
            ({"use_turboquant": True}, "mutually exclusive"),
            ({"max_prefill_chunk": 0}, "max_prefill_chunk"),
            ({"max_prefill_chunk": CONFIG.max_seq_len + 1}, "max_prefill_chunk"),
        )
        with (
            patch.object(export_solo, "_export_mlx") as lower_mlx,
            patch.object(export_solo, "_export_cuda") as lower_cuda,
        ):
            for override, error in cases:
                kwargs = {"backend": "mlx", "max_prefill_chunk": 8}
                kwargs.update(override)
                with self.subTest(**override), self.assertRaisesRegex(
                    ValueError, error
                ):
                    export_solo.export_and_lower(
                        None, CONFIG, ".", use_offgraph_kv_cache=True, **kwargs
                    )
            lower_mlx.assert_not_called()
            lower_cuda.assert_not_called()

    def test_reference_parity_continuation_and_ring_wrap(self):
        for fused in (False, True):
            config = replace(CONFIG, fuse_qkv=fused)
            original = _model(config)
            transformed = copy.deepcopy(original)
            mlx.mlx_source_transformations(
                transformed, dtype=torch.float32, use_offgraph_kv_cache=True
            )
            REGISTRY.install("mg_test", _cache(config))
            # Cross the original ring's physical boundary with chunked prefill
            # and then single-token decode. Use identical fp32 embeddings.
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

    def test_packed_disjoint_positions_match_independent_sessions(self):
        model = _model()
        mlx.mlx_source_transformations(
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

    def test_rope_uses_every_position_in_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check_rope(dtype)

    def _check_rope(self, dtype):
        model = _model().to(dtype)
        mlx.mlx_source_transformations(model, dtype=dtype, use_offgraph_kv_cache=True)
        model.to(dtype)
        attn = model.layers[0].self_attn
        self.assertEqual(attn.inv_freq.dtype, torch.float32)
        x = torch.randn(1, 4, CONFIG.dim, dtype=dtype)
        positions = torch.tensor([10000, 10001, 2, 3])
        captured = {}

        def attend(q, k, v, position, layer_id, scale, dtype):
            captured.update(
                q=q,
                k=k,
                v=v,
                positions=position,
                layer=layer_id,
                scale=scale,
                dtype=dtype,
            )
            return torch.zeros_like(q)

        with patch.object(
            mlx, "update_and_attend", side_effect=attend
        ), torch.no_grad():
            attn(x, positions)
            qkv = attn.qkv_proj(attn.qkv_proj_norm(x))
            q = attn.q_norm(
                qkv[..., : attn.q_dim].reshape(1, 4, attn.n_heads, attn.head_dim)
            ).transpose(1, 2)
            k = attn.k_norm(
                qkv[..., attn.q_dim : attn.q_dim + attn.kv_dim].reshape(
                    1, 4, attn.n_kv_heads, attn.head_dim
                )
            ).transpose(1, 2)
            frequencies = 1.0 / (
                CONFIG.rope_theta
                ** (torch.arange(0, CONFIG.head_dim, 2).float() / CONFIG.head_dim)
            )
            phase = positions.float().outer(frequencies)[None, None]
            expected_q, expected_k = apply_rotary_emb(q, k, phase.cos(), phase.sin())
        torch.testing.assert_close(captured["q"], expected_q, rtol=0, atol=0)
        torch.testing.assert_close(captured["k"], expected_k, rtol=0, atol=0)
        torch.testing.assert_close(captured["positions"], positions[:, None])
        self.assertEqual(captured["q"].shape, (1, CONFIG.n_heads, 4, CONFIG.head_dim))
        for name in ("k", "v"):
            self.assertEqual(
                captured[name].shape, (1, CONFIG.n_kv_heads, 4, CONFIG.head_dim)
            )
        for name in ("q", "k", "v"):
            self.assertEqual(captured[name].dtype, dtype)
        self.assertEqual(captured["layer"], 0)
        self.assertEqual(captured["scale"], attn.attn_scale)
        self.assertEqual(captured["dtype"], dtype)

    def test_selected_rows_before_lm_head_and_dynamic_export(self):
        model = _model()
        mlx.mlx_source_transformations(
            model, dtype=torch.float32, use_offgraph_kv_cache=True
        )
        tokens = torch.tensor([[1, 2, 3, 4]])
        positions = torch.arange(4)
        rows = torch.tensor([3, 0, 3])
        embeddings = model.mlx_embed_text(tokens)
        cache = _cache()
        REGISTRY.install("mg_test", cache)
        with REGISTRY.active("mg_test"), torch.no_grad():
            full = model.mlx_prefill_forward(embeddings, positions, torch.arange(4))
            cache.reset()
            shapes = []
            handle = model.lm_head.register_forward_pre_hook(
                lambda _, args: shapes.append(args[0].shape)
            )
            selected = model.mlx_prefill_forward(embeddings, positions, rows)
            handle.remove()
        self.assertEqual(shapes, [torch.Size([1, 3, CONFIG.dim])])
        torch.testing.assert_close(selected, full[:, rows])
        self.assertEqual(selected.dtype, torch.float32)
        dim = torch.export.Dim("tokens", min=1, max=8)
        selected_dim = torch.export.Dim("rows", min=1, max=8)
        with common.BoundMethodForward(model, model.mlx_prefill_forward):
            program = torch.export.export(
                model,
                (embeddings, positions, rows),
                dynamic_shapes=({1: dim}, {0: dim}, {0: selected_dim}),
                strict=True,
            )
        ops = [
            node
            for node in program.graph.nodes
            if node.target == torch.ops.kvcache.update_and_attend.default
        ]
        self.assertEqual([node.args[4] for node in ops], list(range(CONFIG.n_layers)))
        self.assertFalse(program.graph_signature.buffers_to_mutate)
        self.assertFalse(
            any("cache" in name for name in program.graph_signature.buffers)
        )
        self.assertEqual(len(program.graph_signature.user_inputs), 3)
        self.assertEqual(len(program.graph_signature.user_outputs), 1)
        cache.reset()
        with REGISTRY.active("mg_test"), torch.no_grad():
            got = program.module()(embeddings[:, :1], positions[:1], torch.tensor([0]))
        self.assertEqual(got.shape, (1, 1, CONFIG.vocab_size))

    def test_solo_export_shapes_include_decode_and_bound_both_axes(self):
        class StopBeforeLowering(Exception):
            pass

        # Exercise the actual solo export examples/specs, including the static
        # width-one case and the smallest nonspecialized dynamic example.
        for max_width in (1, 2, 8):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(max_width=max_width, dtype=dtype):
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
                            activation_dtype=dtype,
                            max_prefill_chunk=max_width,
                            use_offgraph_kv_cache=True,
                        )
                    self.assertEqual(
                        set(programs), {"embed_text", "forward_from_embeddings"}
                    )
                    embed = programs["embed_text"].module()
                    forward = programs["forward_from_embeddings"].module()
                    cache = _cache()
                    REGISTRY.install("mg_test", cache)
                    with REGISTRY.active("mg_test"), torch.no_grad():
                        for width, rows in (
                            (1, 1),
                            (max_width, 1),
                            (1, max_width),
                            (max_width, max_width),
                        ):
                            cache.reset()
                            embeddings = embed(
                                torch.zeros((1, width), dtype=torch.long)
                            )
                            self.assertEqual(embeddings.shape, (1, width, CONFIG.dim))
                            self.assertEqual(embeddings.dtype, dtype)
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
                                torch.zeros((1, 1, CONFIG.dim), dtype=dtype),
                                torch.zeros(1, dtype=torch.long),
                                torch.zeros(max_width + 1, dtype=torch.long),
                            )

    def test_metadata_is_opt_in_and_geometry_uses_actual_config(self):
        config = replace(CONFIG, n_layers=5, global_attn_cfg="[3,7,5,0]")
        kwargs = {
            "config": config,
            "max_prefill": 8,
            "activation_dtype": torch.float16,
            "mutable_buffer_metadata": "legacy",
            "has_vision": True,
            "max_vision_patches": 256,
        }
        legacy = export_solo._solo_constant_methods(**kwargs)
        self.assertEqual(legacy["get_activation_dtype"], "float16")
        self.assertEqual(legacy["get_max_seq_len"], config.max_seq_len)
        self.assertEqual(legacy["get_mutable_buffer_metadata"], "legacy")
        self.assertNotIn("get_logits_to_keep_mode", legacy)
        self.assertNotIn("get_max_context_len", legacy)
        for dtype, scalar in (
            (torch.float16, 5),
            (torch.bfloat16, 15),
        ):
            kwargs["activation_dtype"] = dtype
            metadata = export_solo._solo_constant_methods(
                **kwargs, use_offgraph_kv_cache=True
            )
            self.assertEqual(metadata["get_activation_dtype"], int(scalar))
            self.assertEqual(metadata["get_logits_to_keep_mode"], 2)
            self.assertEqual(metadata["get_max_seq_len"], 8)
            self.assertEqual(metadata["get_max_prefill_chunk"], 8)
            self.assertEqual(metadata["get_max_context_len"], config.max_seq_len)
            self.assertEqual(metadata["get_vocab_size"], config.vocab_size)
            self.assertEqual(metadata["get_n_caches"], config.n_layers)
            self.assertEqual(
                metadata["get_windows"].tolist(),
                [config.layer_window_size(i) for i in range(config.n_layers)],
            )
            self.assertEqual(
                metadata["get_kv_heads"].tolist(), [config.n_kv_heads] * config.n_layers
            )
            self.assertEqual(
                metadata["get_head_dims"].tolist(), [config.head_dim] * config.n_layers
            )
            self.assertNotIn("get_mutable_buffer_metadata", metadata)
            self.assertEqual(metadata["get_vision_hidden_size"], config.dim)
            self.assertEqual(metadata["get_max_vision_patches"], 256)
            for name in ("get_windows", "get_kv_heads", "get_head_dims"):
                self.assertEqual(metadata[name].dtype, torch.int32)
        kwargs["activation_dtype"] = torch.float32
        with self.assertRaisesRegex(ValueError, "float16 or bfloat16"):
            export_solo._solo_constant_methods(**kwargs, use_offgraph_kv_cache=True)

    def test_legacy_forward_still_returns_last_logits(self):
        model = _model()
        mlx.mlx_source_transformations(model, dtype=torch.float32, max_write_len=4)
        self.assertTrue(
            all(hasattr(layer.self_attn, "kv_cache") for layer in model.layers)
        )
        with (
            patch.object(
                mlx, "update_and_attend", side_effect=AssertionError("off-graph op")
            ),
            patch.object(torch.ops.mlx, "rope", wraps=torch.ops.mlx.rope) as rope,
            torch.no_grad(),
        ):
            result = model(torch.tensor([[1, 2, 3]]), torch.arange(3))
            decoded = model(torch.tensor([[4]]), torch.tensor([3]))
        self.assertEqual(result.shape, (1, CONFIG.vocab_size))
        self.assertEqual(decoded.shape, (1, CONFIG.vocab_size))
        rope_calls_per_step = 2 * sum(
            layer.self_attn.use_rope for layer in model.layers
        )
        self.assertEqual(
            [call.args[2] for call in rope.call_args_list],
            [0] * rope_calls_per_step + [3] * rope_calls_per_step,
        )

    def test_native_mlx_cache_execution_when_runner_is_built(self):
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes
        from executorch.backends.mlx.test.test_utils import (
            find_op_test_runner,
            load_tensors_from_bin,
            run_cpp_test_runner,
            save_tensors_to_bin,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        try:
            find_op_test_runner()
        except FileNotFoundError as error:
            self.skipTest(str(error))
        model = _model().to(torch.float16)
        mlx.mlx_source_transformations(
            model, dtype=torch.float16, use_offgraph_kv_cache=True
        )
        # The stock runner installs a flat cache. A fresh chunk shorter than
        # every window has identical visibility for the mixed-policy model.
        with torch.no_grad():
            inputs = (
                model.mlx_embed_text(torch.tensor([[1, 2, 3]])),
                torch.arange(3),
                torch.tensor([2, 0]),
            )
            REGISTRY.install("mg_test", _cache())
            with REGISTRY.active("mg_test"):
                expected = model.mlx_prefill_forward(*inputs)
            with common.BoundMethodForward(model, model.mlx_prefill_forward):
                exported = torch.export.export(model, inputs, strict=True)
        edge = to_edge_transform_and_lower(
            exported,
            transform_passes=get_default_passes(),
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common.save_pte(edge.to_executorch(), directory, None)
            save_tensors_to_bin(list(inputs), root / "input.bin")
            self.assertTrue(
                run_cpp_test_runner(
                    root / "model.pte",
                    root / "input.bin",
                    root / "output.bin",
                    kv_cache=f"{CONFIG.max_seq_len},{CONFIG.n_layers},{CONFIG.n_kv_heads},{CONFIG.head_dim},{int(ScalarType.HALF)}",
                    timeout=60,
                )
            )
            actual = load_tensors_from_bin(root / "output.bin")[0]
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)

    def test_real_mlx_lowering(self):
        from executorch.backends.mlx import preprocess
        from executorch.backends.mlx.serialization.mlx_graph_schema import (
            UpdateAndAttendNode,
        )
        from executorch.runtime import Runtime, Verification

        graphs = []
        serialize = preprocess.serialize_mlx_graph

        def capture_graph(graph):
            graphs.append(graph)
            return serialize(graph)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            preprocess, "serialize_mlx_graph", side_effect=capture_graph
        ):
            export_solo.export_and_lower(
                _model(),
                CONFIG,
                directory,
                backend="mlx",
                activation_dtype=torch.float16,
                max_prefill_chunk=8,
                use_offgraph_kv_cache=True,
            )
            program = Runtime.get().load_program(
                directory + "/model.pte", verification=Verification.Minimal
            )
            self.assertTrue(
                {"embed_text", "forward_from_embeddings"}.issubset(program.method_names)
            )
            self.assertNotIn("decode_from_embedding", program.method_names)
            self.assertNotIn("get_mutable_buffer_metadata", program.method_names)
            self.assertEqual(
                program.load_method("get_logits_to_keep_mode").execute([]), [2]
            )
            self.assertEqual(
                program.load_method("get_activation_dtype").execute([]),
                [int(ScalarType.HALF)],
            )
            self.assertEqual(program.load_method("get_max_seq_len").execute([]), [8])
            self.assertEqual(
                program.load_method("get_max_context_len").execute([]),
                [CONFIG.max_seq_len],
            )
        # Both methods are fully delegated; no portable frequency-compute island.
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
        self.assertTrue(all(graph.num_mutable_buffer_tensors == 0 for graph in graphs))


if __name__ == "__main__":
    # run.py uses runpy without installing this module as sys.modules["__main__"].
    import sys

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MLXOffgraphTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
