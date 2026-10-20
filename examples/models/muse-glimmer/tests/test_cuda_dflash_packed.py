# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent tiny draft math and packed export shape checks, without MG weights."""

import copy
import importlib.resources
import json
import unittest
import unittest.mock
from dataclasses import replace
from pathlib import Path

import torch
from executorch.examples.models.muse_glimmer.export import common
from executorch.examples.models.muse_glimmer.export.export_dflash import (
    cuda_owner_verifier_programs,
    prepare_cuda_packed_draft,
)
from executorch.examples.models.muse_glimmer.export.export_solo import step_forward
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    DFlashConfig,
    DFlashDraftModel,
    MuseGlimmerWithDFlash,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
    add_dflash_hidden_tapping,
    enable_dflash_offgraph_kv_cache,
)
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG
from executorch.extension.llm.cache.reference_cache import AttendSpec, MaskKind
from executorch.extension.llm.cache.update_and_attend import REGISTRY


def models(block=4, head_dim=8):
    torch.manual_seed(123)
    tc = replace(TINY_CONFIG, dim=32, n_layers=2, head_dim=head_dim, max_seq_len=128)
    target = MuseGlimmerModel(tc).to(torch.bfloat16).eval()
    target.activation_dtype = torch.bfloat16
    dc = DFlashConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=head_dim,
        ffn_dim=64,
        vocab_size=tc.vocab_size,
        target_layers=[0, 1],
        block_size=block,
        mask_token_id=255,
        max_seq_len=128,
    )
    draft = DFlashDraftModel(dc, max_context_length=64).to(torch.bfloat16).eval()
    return target, draft


class PackedOracleCache:
    """Test-only grouped noncausal cache: retain context, discard every block."""

    def __init__(self, contexts, block):
        self.contexts = contexts
        self.block = block
        self.history = {}
        self.calls = []

    def update_and_fetch(self, layer, k, v, positions):
        self.calls.append((k.shape[2], positions.clone()))
        specs = []
        at = 0
        for owner, context in enumerate(self.contexts):
            end = at + context + self.block
            new_k, new_v = k[:, :, at:end], v[:, :, at:end]
            old = self.history.get((layer, owner))
            hk = new_k if old is None else torch.cat([old[0], new_k], dim=2)
            hv = new_v if old is None else torch.cat([old[1], new_v], dim=2)
            self.history[layer, owner] = (
                hk[:, :, : -self.block].clone(),
                hv[:, :, : -self.block].clone(),
            )
            specs.append(AttendSpec(hk, hv, MaskKind.NONE, self.block))
            at = end
        if at != k.shape[2]:
            raise ValueError("oracle grouping does not cover K/V")
        return specs


class PackedDraftMathTest(unittest.TestCase):
    def test_uneven_context_nonidentity_order_and_zero_new_context(self):
        target, draft = models()
        refs = [copy.deepcopy(draft), copy.deepcopy(draft)]
        enable_dflash_offgraph_kv_cache(draft, 32)
        combined = MuseGlimmerWithDFlash(target, draft, target.config, draft.config)
        cache = PackedOracleCache([3, 1], 4)
        key = "tiny-dflash-packed"
        REGISTRY.install(key, cache)
        self.addCleanup(REGISTRY.uninstall, key)
        starts = [0, 0]
        for contexts in ([3, 1], [0, 0], [1, 2]):
            cache.contexts = contexts
            tokens = torch.randint(0, 32, (2, 4))
            hidden = torch.randn(1, sum(contexts), 64, dtype=torch.bfloat16)
            ctx_pos, block_pos, order = [], [], []
            offset = 0
            expected = []
            for owner, count in enumerate(contexts):
                ctx_pos.extend(range(starts[owner], starts[owner] + count))
                block_pos.extend(
                    range(starts[owner] + count, starts[owner] + count + 4)
                )
                order.extend(range(offset, offset + count))
                order.extend(
                    range(sum(contexts) + owner * 4, sum(contexts) + (owner + 1) * 4)
                )
                with torch.no_grad():
                    emb = target.embed_tokens(tokens[owner : owner + 1]).to(
                        torch.bfloat16
                    )
                    h = refs[owner](
                        emb,
                        hidden[:, offset : offset + count],
                        torch.tensor([starts[owner]]),
                    )
                    expected.append(target.lm_head(h).float())
                offset += count
                starts[owner] += count
            pos = torch.tensor(ctx_pos + block_pos)
            with torch.no_grad(), REGISTRY.active(key):
                actual = combined.draft_forward_packed(
                    tokens, hidden, pos, torch.tensor(order)
                )
            torch.testing.assert_close(
                actual, torch.cat(expected, dim=1), atol=0.025, rtol=0.025
            )
            self.assertEqual(cache.calls[-1][0], sum(contexts) + 8)
            torch.testing.assert_close(cache.calls[-1][1][:, 0], pos[order])
            self.assertEqual(tuple(actual.shape), (1, 8, target.config.vocab_size))

    def test_taps_keep_every_row_and_selected_logits_match(self):
        target, _ = models()
        ordinary = copy.deepcopy(target)
        add_dflash_hidden_tapping(target, [0, 1])
        captured = []
        handles = [
            layer.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].clone())
            )
            for layer in ordinary.layers
        ]
        try:
            tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
            positions = torch.arange(6)
            with torch.no_grad():
                full = ordinary(tokens, positions)
                logits, taps = step_forward(
                    target, tokens, positions, torch.tensor([5, 1, 5])
                )
            torch.testing.assert_close(logits, full[0, [5, 1, 5]])
            torch.testing.assert_close(taps, torch.cat(captured, dim=-1))
            self.assertEqual(tuple(taps.shape), (1, 6, 64))
        finally:
            for handle in handles:
                handle.remove()

    def test_invalid_tap_ids_fail_before_binding(self):
        for ids in ([], [1, 0], [0, 0], [-1], [2], [True]):
            target, _ = models()
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                add_dflash_hidden_tapping(target, ids)
            self.assertFalse(hasattr(target, "_dflash_target_layer_ids"))

    def test_invalid_packed_limits_fail_before_transform(self):
        for overrides in (
            {"max_owners": True},
            {"max_owners": 0},
            {"max_owners": 5},
            {"max_step_tokens": 1},
            {"max_cells": 80},
        ):
            target, draft = models(block=16)
            options = {"max_step_tokens": 16, "max_cells": 128, "max_owners": 4}
            options.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                prepare_cuda_packed_draft(target, target.config, draft, **options)
            self.assertFalse(hasattr(target, "_dflash_target_layer_ids"))
            self.assertTrue(hasattr(draft.layers[0].self_attn, "kv_cache"))

    def test_single_owner_export_and_malformed_width(self):
        target, draft = models(block=16)
        programs, _, metadata = prepare_cuda_packed_draft(
            target,
            target.config,
            draft,
            max_step_tokens=16,
            max_cells=128,
            max_owners=1,
        )
        self.assertEqual(metadata["get_draft_max_step_tokens"], 32)
        module = programs["draft_forward"].module()
        key = "tiny-single-owner"
        REGISTRY.install(key, PackedOracleCache([0], 16))
        self.addCleanup(REGISTRY.uninstall, key)
        args = (
            torch.zeros(1, 16, dtype=torch.long),
            torch.zeros(1, 0, 64, dtype=torch.bfloat16),
            torch.arange(16),
            torch.arange(16),
        )
        with torch.no_grad(), REGISTRY.active(key):
            self.assertEqual(tuple(module(*args).shape), (1, 16, 256))
        with self.assertRaises((RuntimeError, AssertionError)):
            module(args[0], args[1], torch.arange(17), torch.arange(17))

    def test_dynamic_export_matches_static_at_zero_and_unequal_context(self):
        target, draft = models(block=16)
        programs, manifest, metadata = prepare_cuda_packed_draft(
            target,
            target.config,
            draft,
            max_step_tokens=16,
            max_cells=128,
            max_owners=4,
        )
        self.assertEqual(json.loads(manifest)["max_write"], 80)
        self.assertEqual(metadata["get_block_size"], 16)
        self.assertEqual(metadata["get_draft_max_owners"], 4)
        self.assertEqual(metadata["get_draft_max_step_tokens"], 80)
        state = programs["draft_forward"].state_dict
        self.assertIs(state["embed_tokens.weight"], target.embed_tokens.weight)
        self.assertIs(state["lm_head.weight"], target.lm_head.weight)
        self.assertFalse(any(name.startswith("target.") for name in state))
        self.assertEqual(metadata["get_draft_n_caches"], 2)
        self.assertNotEqual(
            metadata["get_dflash_target_cache_key"],
            metadata["get_dflash_draft_cache_key"],
        )
        combined = MuseGlimmerWithDFlash(target, draft, target.config, draft.config)
        key = "tiny-dynamic-draft"
        for owners, context in ((1, 0), (1, 1), (2, 3), (3, 7), (4, 1), (4, 16)):
            q = owners * 16
            args = (
                torch.zeros(owners, 16, dtype=torch.long),
                torch.randn(1, context, 64, dtype=torch.bfloat16),
                torch.arange(context + q),
                torch.arange(context + q),
            )
            with (
                common.BoundMethodForward(combined, combined.draft_forward_packed),
                torch.no_grad(),
            ):
                static = torch.export.export(combined, args, strict=True)
            counts = [context] + [0] * (owners - 1)
            results = []
            for module in (programs["draft_forward"].module(), static.module()):
                REGISTRY.install(key, PackedOracleCache(counts, 16))
                try:
                    with REGISTRY.active(key), torch.no_grad():
                        results.append(module(*args))
                finally:
                    REGISTRY.uninstall(key)
            torch.testing.assert_close(*results)


class PackedDraftCudaTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_exported_lowered_model_numerics_all_owner_counts(self):
        from executorch.backends.cuda.passes.lower_offgraph_kv import (
            CheckOffGraphKVStepWidthPass,
            LowerOffGraphKVPass,
            OFFGRAPH_KV_CELLS_FQN,
            offgraph_kv_mask_fqn,
            OFFGRAPH_KV_READ_LEN_FQN,
            parse_offgraph_kv_manifest,
        )
        from executorch.exir import EdgeCompileConfig, to_edge

        target, draft = models(block=16, head_dim=16)
        programs, manifest, _ = prepare_cuda_packed_draft(
            target,
            target.config,
            draft,
            max_step_tokens=16,
            max_cells=128,
            max_owners=4,
        )
        edge = to_edge(
            programs["draft_forward"],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()
        CheckOffGraphKVStepWidthPass(b"2:0", layout="cell")(edge)
        lowered = LowerOffGraphKVPass(parse_offgraph_kv_manifest(manifest.encode()))(
            edge
        )
        from executorch.backends.cuda.cuda_backend import _move_graph_nodes_to_device

        # Lowering deliberately frees runtime-owned cache storage for compilation.
        # The numerical harness supplies actual storage before the device move.
        for name, value in list(lowered.constants.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.untyped_storage().nbytes() == 0
            ):
                lowered.constants[name] = torch.zeros(value.shape, dtype=value.dtype)
        # The lowered graph is intentionally mutable, so use CUDA's rewrite
        # rather than the upstream functional-ATen export verifier.
        _move_graph_nodes_to_device(lowered.graph_module, torch.device("cuda"))
        oracle = MuseGlimmerWithDFlash(
            target, draft, target.config, draft.config
        ).cuda()
        key = "tiny-lowered-model"
        for counts in (
            [0],
            [3, 1],
            [0, 2, 4],
            [0, 0, 0, 0],
            [1, 0, 3, 2],
            [4, 4, 4, 4],
        ):
            context, queries = sum(counts), 16 * len(counts)
            order, ctx_pos, block_pos = [], [], []
            offset = 0
            mask = torch.zeros(1, 1, 80, 128, dtype=torch.bool, device="cuda")
            grouped = 0
            for owner, count in enumerate(counts):
                ctx_pos.extend(range(count))
                block_pos.extend(range(count, count + 16))
                order.extend(range(offset, offset + count))
                order.extend(range(context + owner * 16, context + (owner + 1) * 16))
                mask[
                    :, :, owner * 16 : (owner + 1) * 16, grouped : grouped + count + 16
                ] = True
                offset += count
                grouped += count + 16
            args = (
                torch.randint(0, 256, (len(counts), 16), device="cuda"),
                torch.randn(1, context, 64, dtype=torch.bfloat16, device="cuda"),
                torch.tensor(ctx_pos + block_pos, device="cuda"),
                torch.tensor(order, device="cuda"),
            )
            lowered.constants[OFFGRAPH_KV_CELLS_FQN] = torch.arange(80, device="cuda")
            lowered.constants[OFFGRAPH_KV_READ_LEN_FQN] = torch.tensor(
                [grouped], device="cuda"
            )
            for layer in range(2):
                lowered.constants[f"__et_offgraph_kv_layer_{layer}_k"] = torch.zeros(
                    1, 128, 2, 16, dtype=torch.bfloat16, device="cuda"
                )
                lowered.constants[f"__et_offgraph_kv_layer_{layer}_v"] = torch.zeros(
                    1, 128, 2, 16, dtype=torch.bfloat16, device="cuda"
                )
                lowered.constants[offgraph_kv_mask_fqn(0)] = mask
            module = lowered.module().cuda()
            REGISTRY.install(key, PackedOracleCache(counts, 16))
            try:
                with torch.no_grad(), REGISTRY.active(key):
                    expected = oracle.draft_forward_packed(*args)
                    actual = module(*args)
                torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.035)
                self.assertEqual(tuple(actual.shape), (1, queries, 256))
            finally:
                REGISTRY.uninstall(key)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cell_step_writes_k_width_and_reads_only_block_queries(self):
        from executorch.backends.cuda.passes.lower_offgraph_kv import cell_step

        torch.manual_seed(34)
        q = torch.randn(1, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 12, 16, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cells = torch.tensor([11, 3, 8, 0, 9, 2, 4, 10, 1, 7, 5, 6], device="cuda")
        mask = torch.zeros(1, 1, 16, 20, device="cuda", dtype=torch.bool)
        mask[:, :, :4, cells[:7]] = True
        mask[:, :, 4:8, cells[7:]] = True
        kp = torch.zeros(1, 20, 2, 16, device="cuda", dtype=torch.bfloat16)
        vp = torch.zeros_like(kp)
        actual = cell_step(
            q, k, v, kp, vp, cells, torch.tensor([12], device="cuda"), mask, 16**-0.5
        )
        expected = []
        for qs, ks in ((slice(0, 4), slice(0, 7)), (slice(4, 8), slice(7, 12))):
            expected.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, qs].float(),
                    k[:, :, ks].float().repeat_interleave(2, 1),
                    v[:, :, ks].float().repeat_interleave(2, 1),
                    scale=16**-0.5,
                ).to(torch.bfloat16)
            )
        torch.testing.assert_close(actual, torch.cat(expected, 2), atol=0.01, rtol=0.01)
        torch.testing.assert_close(kp[:, cells].transpose(1, 2), k)
        torch.testing.assert_close(vp[:, cells].transpose(1, 2), v)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_tiny_combined_lowering_schema(self):
        import tempfile

        from executorch.examples.models.muse_glimmer.export.export_solo import (
            export_cuda_batching,
        )
        from executorch.exir.lowered_backend_module import get_lowered_submodules

        target, draft = models(block=16, head_dim=16)
        source = Path(__file__).resolve().parents[4]
        read_resource = importlib.resources.read_binary

        def owned_schema(package, name):
            # Source-only launchers omit the wheel's copied schema resources.
            if package == "executorch.exir._serialize" and name in (
                "program.fbs",
                "scalar_type.fbs",
            ):
                return (source / "schema" / name).read_bytes()
            return read_resource(package, name)

        with (
            tempfile.TemporaryDirectory() as out,
            unittest.mock.patch(
                "importlib.resources.read_binary", side_effect=owned_schema
            ),
        ):
            program = export_cuda_batching(
                target,
                target.config,
                out,
                max_step_tokens=16,
                max_cells=128,
                forward_sizes=(1, 2, 4, 8),
                dflash_draft=draft,
                max_draft_owners=4,
            )
            self.assertGreater((Path(out) / "model.pte").stat().st_size, 0)
            self.assertTrue(list(Path(out).glob("*.ptd")))
        self.assertEqual(
            set(program.methods),
            {
                "forward_1",
                "forward_2",
                "forward_4",
                "forward_8",
                "forward_others",
                "draft_forward",
                "sample",
                "sample_argmax",
                "dflash_verify",
                "dflash_verify_argmax",
            },
        )
        from executorch.backends.cuda.cuda_weight_collector import (
            decode_cuda_aoti_metadata,
        )

        shared_keys = []
        for name in ("forward_4", "forward_others", "draft_forward"):
            lowered = get_lowered_submodules(
                program.exported_program(name).graph_module
            )
            self.assertEqual(len(lowered), 1)
            specs = {s.key: s.value for s in lowered[0][1].compile_specs}
            self.assertEqual(
                specs["offgraph_kv_step_width"],
                b"2:0" if name == "draft_forward" else b"1:0",
            )
            self.assertEqual(
                specs["llm_cache_registry_key"],
                (
                    b"muse_glimmer_draft"
                    if name == "draft_forward"
                    else b"muse_glimmer_target"
                ),
            )
            self.assertEqual(specs["low_memory_mode"], b"ON")
            self.assertNotIn("autotune_at_compile_time", specs)
            weights = decode_cuda_aoti_metadata(lowered[0][1].processed_bytes)
            shared_keys.append(
                {
                    entry.fqn: entry.storage_key
                    for entry in weights.entries
                    if entry.fqn in ("embed_tokens.weight", "lm_head.weight")
                }
            )
        self.assertEqual(set(shared_keys[0]), {"embed_tokens.weight", "lm_head.weight"})
        self.assertEqual(shared_keys[0], shared_keys[1])
        self.assertEqual(shared_keys[0], shared_keys[2])


class SingleOwnerVerifierExportTest(unittest.TestCase):
    def test_single_owner_accepts_runtime_draft_widths(self):
        methods = cuda_owner_verifier_programs(32, 1, 16)
        verifier = methods["dflash_verify_argmax"].module()
        for width in (3, 4, 16):
            logits = torch.zeros(1, width, 32)
            logits[:, :, 7] = 1
            candidates = torch.full((1, width), 7, dtype=torch.long)
            self.assertEqual(verifier(logits, candidates).tolist(), [[width - 1, 7]])
            candidates[0, 2] = 5
            self.assertEqual(verifier(logits, candidates).tolist(), [[1, 7]])


if __name__ == "__main__":
    unittest.main()
