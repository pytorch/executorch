# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.lfm3.convert_weights import lfm_3_to_meta
from executorch.examples.models.lfm3.test_utils import (
    DIM,
    FULL_THETA,
    generic_model,
    HEAD_DIM,
    LAYER_TYPES,
    random_hf_state_dict,
    run,
    shared_meta_config,
    tiny_hf_config,
    tiny_meta_config,
    VOCAB,
    WINDOW,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.extension.llm.export.builder import LLMEdgeManager


class Lfm3GenericTest(unittest.TestCase):
    def test_dynamic_prefill_metadata_preserves_context_contract(self) -> None:
        class WithPosition(torch.nn.Module):
            def forward(self, tokens, args):
                return tokens + args["input_pos"][0]

        for use_kv_cache in (False, True):
            with self.subTest(use_kv_cache=use_kv_cache):
                tokens = torch.ones(1, 2)
                model = WithPosition() if use_kv_cache else torch.nn.Identity()
                inputs = (
                    (tokens, {"input_pos": torch.tensor([0])})
                    if use_kv_cache
                    else (tokens,)
                )
                manager = LLMEdgeManager(
                    model=model,
                    modelname="prefill_bound",
                    max_seq_len=8,
                    use_kv_cache=use_kv_cache,
                    example_inputs=inputs,
                    enable_dynamic_shape=True,
                )
                exported = manager._export()
                self.assertEqual(manager.metadata["get_max_seq_len"], 8)
                self.assertTrue(exported.range_constraints)
                self.assertEqual(
                    int(next(iter(exported.range_constraints.values())).upper),
                    7,
                )

    def test_static_prefill_metadata_keeps_fixed_length(self) -> None:
        manager = LLMEdgeManager(
            model=torch.nn.Identity(),
            modelname="static_prefill_bound",
            max_seq_len=8,
            use_kv_cache=False,
            example_inputs=(torch.ones(1, 8),),
        )
        self.assertEqual(manager.metadata["get_max_seq_len"], 8)
        self.assertFalse(manager._export().range_constraints)

    def test_theta_identity(self) -> None:
        rope = shared_meta_config()["rope_parameters"]["full_attention"]
        rotary_dim = int(HEAD_DIM * rope["partial_rotary_factor"])
        for index in range(rotary_dim // 2):
            self.assertAlmostEqual(
                rope["rope_theta"] ** (-2 * index / rotary_dim),
                FULL_THETA ** (-2 * index / HEAD_DIM),
            )

    def test_shared_converter_permutation_and_projection_splits(self) -> None:
        raw = random_hf_state_dict()
        converted = lfm_3_to_meta(raw, tiny_hf_config())
        # Independent expected ordering for head_dim=16, rotary_dim=8.
        permutation = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        for layer, layer_type in enumerate(LAYER_TYPES):
            hf_prefix = f"model.layers.{layer}."
            meta_prefix = f"layers.{layer}."
            if layer_type == "conv":
                splits = raw[hf_prefix + "conv.in_proj.weight"].split(DIM, dim=0)
                for name, value in zip(("B_proj", "C_proj", "x_proj"), splits):
                    torch.testing.assert_close(
                        converted[meta_prefix + f"conv.{name}.weight"],
                        value,
                        rtol=0,
                        atol=0,
                    )
                self.assertNotIn(meta_prefix + "conv.in_proj.weight", converted)
                continue
            for hf_name, meta_name in (("q_proj", "wq"), ("k_proj", "wk")):
                weight = raw[hf_prefix + f"self_attn.{hf_name}.weight"]
                expected = weight
                if layer_type == "full_attention":
                    expected = weight.view(-1, HEAD_DIM, DIM)[:, permutation].reshape(
                        weight.shape
                    )
                torch.testing.assert_close(
                    converted[meta_prefix + f"attention.{meta_name}.weight"],
                    expected,
                    rtol=0,
                    atol=0,
                )
            for name in ("q", "k"):
                weight = raw[hf_prefix + f"self_attn.{name}_norm.weight"]
                expected = (
                    weight[permutation] if layer_type == "full_attention" else weight
                )
                torch.testing.assert_close(
                    converted[meta_prefix + f"attention.{name}_norm_fn.weight"],
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_default_converter_uses_shared_config(self) -> None:
        config = shared_meta_config()
        head_dim = config["head_dim"]
        layer = config["layer_types"].index("full_attention")
        raw = {
            "model.embed_tokens.weight": torch.ones(3, 4),
            f"model.layers.{layer}.self_attn.q_proj.weight": torch.arange(head_dim * 4)
            .view(head_dim, 4)
            .float(),
        }
        default = lfm_3_to_meta(raw)
        explicit = lfm_3_to_meta(raw, config)
        self.assertEqual(set(default), set(explicit))
        for key in default:
            torch.testing.assert_close(default[key], explicit[key], rtol=0, atol=0)

    def test_chunked_matches_window_prefill(self) -> None:
        tokens = torch.randint(
            3, VOCAB, (20,), generator=torch.Generator().manual_seed(1)
        ).tolist()
        ref = run(generic_model(), tokens, chunk=WINDOW)
        for chunk in (1, 3, 5):
            with self.subTest(chunk=chunk):
                got = run(generic_model(), tokens, chunk=chunk)
                self.assertTrue(torch.isfinite(got).all())
                torch.testing.assert_close(got, ref, rtol=0, atol=1e-4)

    def test_sliding_prefill_bound(self) -> None:
        for length in (1, WINDOW):
            with self.subTest(length=length):
                generic_model(max_seq_len=length)
        with self.assertRaisesRegex(ValueError, "max_seq_len.*sliding_window"):
            generic_model(max_seq_len=WINDOW + 1)

    def test_no_cache_does_not_apply_ring_prefill_bound(self) -> None:
        config = tiny_meta_config()
        config.update(max_seq_len=WINDOW + 1, max_context_len=32, use_kv_cache=False)
        self.assertIsInstance(
            construct_transformer(ModelArgs(**config)),
            torch.nn.Module,
        )
