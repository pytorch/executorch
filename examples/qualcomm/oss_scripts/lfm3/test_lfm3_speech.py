# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the LFM3-Omni speech pieces of the QNN static flow."""

import math
import unittest
from argparse import Namespace

import torch
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.lfm3.audio import StaticAudioEncoder
from executorch.examples.qualcomm.oss_scripts.lfm3.depth import StaticFrameEmbedding
from executorch.examples.qualcomm.oss_scripts.lfm3.preprocess import audio_features
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    ShortConvHybridLlamaModel,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
    TextDecoder,
)

CTX = 16


def tiny_decoder(ar_len: int, output_hidden: bool) -> ShortConvHybridLlamaModel:
    cfg = ModelArgs(
        dim=32,
        n_layers=3,
        hidden_dim=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        vocab_size=50,
        norm_eps=1e-5,
        use_hf_rope=True,
        use_qk_norm=True,
        qk_norm_before_rope=True,
        headwise_attn_output_gate=True,
        post_attention_norm=True,
        post_ffn_norm=True,
        conv_L_cache=3,
        sliding_window=4,
        layer_types=["conv", "full_attention", "sliding_attention"],
        rope_parameters={
            "full_attention": {"rope_theta": 1000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_theta": 1e4},
        },
        max_context_len=CTX,
        max_seq_len=CTX,
        use_kv_cache=True,
    )
    return ShortConvHybridLlamaModel(
        cfg, ar_len=ar_len, output_hidden=output_hidden
    ).eval()


def decoder_inputs(model: ShortConvHybridLlamaModel):
    ar, cache_len = model.ar_len, CTX - model.ar_len
    mask = torch.zeros(1, 1, ar, CTX)
    return (
        torch.randint(3, 50, (1, ar), dtype=torch.int32),
        mask,
        mask,
        torch.arange(ar, dtype=torch.int32)[None],
        *[torch.zeros(1, 2, 8, cache_len) for _ in model.attention_layers],
        *[torch.zeros(1, 2, cache_len, 8) for _ in model.attention_layers],
        *[torch.zeros(1, model.conv_state_len, 32) for _ in model.conv_layers],
    )


class Lfm3SpeechTest(unittest.TestCase):
    def test_registered_variants_share_decoder_format_and_expand_modality_tokens(self):
        for name, token, count in (
            ("lfm3-2b_vl", "<|image|>", 196),
            ("lfm3-2b_omni", "<|image|>", 196),
            ("lfm3-2b_omni_speech", "<|audio|>", 128),
        ):
            with self.subTest(model=name):
                config = SUPPORTED_LLM_MODELS[name]
                self.assertEqual(
                    config.params_path, SUPPORTED_LLM_MODELS["lfm3-1_7b"].params_path
                )
                self.assertIs(
                    config.convert_weights.__func__,
                    SUPPORTED_LLM_MODELS["lfm3-1_7b"].convert_weights.__func__,
                )
                self.assertIs(LLM_VARIANT_ARCHS[name], ShortConvHybridLlamaModel)
                wrapper = TokenizerWrapper.__new__(TokenizerWrapper)
                wrapper.decoder_model = name
                wrapper.config = config
                wrapper.verbose = False
                expanded = wrapper.prepare_multimodal_prompt(
                    f"<|vision_start|>{token}<|vision_end|>Describe this."
                )
                self.assertEqual(expanded.count(token), count)
                self.assertEqual(expanded.count("<|vision_start|>"), 1)
                self.assertEqual(expanded.count("<|vision_end|>"), 1)

    def test_decoder_returns_hidden_state_after_logits(self):
        torch.manual_seed(0)
        plain = tiny_decoder(ar_len=4, output_hidden=False)
        speech = tiny_decoder(ar_len=4, output_hidden=True)
        speech.load_state_dict(plain.state_dict())
        inputs = decoder_inputs(plain)
        with torch.no_grad():
            ref = plain(*inputs)
            logits, hidden, *caches = speech(*inputs)
        self.assertEqual(len(caches), len(ref) - 1)
        self.assertTrue(torch.equal(logits, ref[0]))
        self.assertTrue(torch.allclose(speech.output(hidden), logits))
        self.assertEqual(speech.get_metadata()["get_output_hidden"], 1)

    def test_hidden_output_tag_does_not_depend_on_conv_state_shape(self):
        graph = torch.fx.Graph()
        logits = graph.placeholder("logits")
        hidden = graph.call_function(torch.ops.aten.neg.default, (logits,))
        hidden.meta["val"] = torch.zeros(1, 1, 32)
        graph.output((logits, hidden))
        decoder = TextDecoder.__new__(TextDecoder)
        decoder.control_args = Namespace(decoder_model="no_conv_layers")
        decoder.meta = {
            "get_output_hidden": 1,
            "get_max_batch_size": 1,
            "get_ar_len": 1,
            "get_max_context_len": CTX,
            "get_head_dim": 8,
        }
        decoder.io_shape = {(1, 1, 50)}
        decoder.kv_cache_shape = {(8, CTX), (CTX, 8), (8, 1), (1, 8)}
        types = {"io_type": torch.uint16, "kv_type": torch.uint8}
        self.assertEqual(decoder._tag_ios(hidden, types), torch.uint16)
        self.assertIsNone(decoder._tag_ios(hidden, types, is_tok_embedding=True))
        decoder.meta["get_output_hidden"] = 0
        self.assertIsNone(decoder._tag_ios(hidden, types))

    def test_hidden_output_quant_attrs_require_dequantization(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("hidden")
        hidden = graph.call_function(
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            (source, 0.5, 13, 0, 65535, torch.uint16),
        )
        output = graph.output((source, hidden))
        decoder = TextDecoder.__new__(TextDecoder)
        decoder.meta = {"get_output_hidden": 1}
        decoder.decoder = torch.fx.GraphModule(torch.nn.Module(), graph)
        decoder._save_hidden_quant_attrs()
        self.assertEqual(decoder.meta["get_hidden_scale"], 0.5)
        self.assertEqual(decoder.meta["get_hidden_zero_point"], 13)
        output.args = ((source, source),)
        with self.assertRaisesRegex(RuntimeError, "Hidden-state output 1"):
            decoder._save_hidden_quant_attrs()

    def test_frame_embedding_sums_one_row_per_codebook(self):
        embed = StaticFrameEmbedding(n_cb=3, codebook_size=5, text_hidden=4)
        with torch.no_grad():
            embed.table.copy_(torch.arange(15 * 4, dtype=torch.float).view(15, 4))
            got = embed(torch.tensor([1, 0, 4], dtype=torch.int32))
        expected = embed.table[[1, 5, 14]].sum(0).view(1, 1, 4)
        self.assertTrue(torch.equal(got, expected))

    def test_audio_features_mark_valid_frames(self):
        t = torch.arange(16000) / 16000
        features = audio_features(torch.sin(2 * math.pi * 440 * t), max_frames=256)
        # HF pads the clip to whole hops: 1 + ceil((samples - n_fft) / hop).
        frames = 1 + -(-(16000 - 400) // 160)
        self.assertEqual(features.shape, (1, 129, 256))
        self.assertEqual(int(features[0, 128].sum()), frames)
        self.assertTrue(torch.all(features[0, :, frames:] == 0))

    def test_audio_encoder_ignores_padding(self):
        torch.manual_seed(0)
        enc_cfg = {
            "num_mel_bins": 16,
            "conv_channels": 8,
            "hidden_size": 32,
            "sliding_window": 4,
            "text_config": {
                "hidden_size": 32,
                "layer_types": ["conv", "full_attention"],
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "norm_eps": 1e-5,
                "conv_L_cache": 3,
                "rope_parameters": {"rope_theta": 1e4},
            },
        }
        encoder = StaticAudioEncoder(
            enc_cfg, text_hidden=24, max_frames=64, hidden_ffn=48
        ).eval()
        valid = 37
        clean = torch.zeros(1, 17, 64)
        clean[0, :16, :valid] = torch.randn(16, valid)
        clean[0, 16, :valid] = 1
        noisy = clean.clone()
        noisy[0, :16, valid:] = torch.randn(16, 64 - valid)
        with torch.no_grad():
            a, b = encoder(clean), encoder(noisy)
        tokens = -(-valid // 8)
        self.assertEqual(a.shape, (1, 8, 24))
        self.assertTrue(torch.allclose(a[:, :tokens], b[:, :tokens], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
