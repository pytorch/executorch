# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The KV-cache-as-I/O rewrite must reproduce the model it rewrites.

A tiny random Llama is exported the way Hugging Face exporters do it (static
cache as inputs), rewritten, and driven the way qnn_llama_runner drives it
(chunked prompt step, then decode steps writing the new slice into the past
buffers under the smart mask). Its greedy tokens must equal the model's own.
"""

import unittest

import torch

try:
    from transformers import (
        GenerationConfig,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from transformers.exporters import DynamoConfig, DynamoExporter
except ImportError:  # transformers without the exporters
    DynamoExporter = None

from executorch.backends.qualcomm.utils import kv_cache_io

L = 32  # context length


def _model():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=L,
    )
    return LlamaForCausalLM(config).eval()


def _qwen3_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=L,
    )
    return Qwen3ForCausalLM(config).eval()


@unittest.skipIf(DynamoExporter is None, "needs transformers.exporters")
class TestKvCacheIo(unittest.TestCase):
    make_model = staticmethod(_model)

    def setUp(self):
        self.model = self.make_model()
        sample = torch.arange(4).reshape(1, 4) + 3
        self.artifacts = artifacts = DynamoExporter().export_for_generation(
            self.model,
            {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
            DynamoConfig(),
            generation_config=GenerationConfig(
                do_sample=False, cache_implementation="static", max_cache_len=L
            ),
        )
        self.prefill = kv_cache_io.to_kv_cache_io(
            artifacts.components["prefill"].artifact
        )
        self.decode = kv_cache_io.to_kv_cache_io(
            artifacts.components["decode"].artifact
        )

    def test_contract(self):
        self.assertEqual(self.decode.method_name, "kv_forward")
        self.assertEqual(self.prefill.method_name, "prefill_forward")
        self.assertEqual(self.decode.k_shapes[0], torch.Size([1, 2, 16, L - 1]))
        self.assertEqual(self.decode.v_shapes[0], torch.Size([1, 2, L - 1, 16]))
        self.assertEqual(tuple(self.prefill.logits_shape), (1, 4, 128))
        self.assertEqual(
            self.decode.constant_methods()["get_io_contract"], "kvcache_io"
        )
        targets = {str(n.target) for n in self.decode.module.graph.nodes}
        self.assertFalse(
            any("index_copy" in t or "scaled_dot_product" in t for t in targets)
        )
        # RoPE runs in halves, without building rotate_half's rotated copy.
        self.assertNotIn("aten.neg.default", targets)

    def test_weight_arithmetic_is_folded(self):
        # Gemma-style RMSNorm scale: (1 + weight) becomes one constant.
        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8))

            def forward(self, x):
                norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return norm * (1.0 + self.weight.float())

        gm = torch.export.export(Norm(), (torch.randn(2, 8),)).module()
        x = torch.randn(2, 8)
        expected = gm(x)
        kv_cache_io.fold_weight_arithmetic(gm)
        gm.recompile()
        last = next(n for n in gm.graph.nodes if n.op == "output").args[0][0]
        self.assertTrue(any(a.op == "get_attr" for a in last.args))
        torch.testing.assert_close(gm(x), expected)
        # The rewritten step re-exports cleanly.
        self.decode.export()

    def test_greedy_matches_model(self):
        self._check_greedy(self.decode, self.prefill, -65535.0)

    def test_greedy_matches_model_masked_softmax(self):
        artifacts = self.artifacts.components
        self._check_greedy(
            kv_cache_io.to_kv_cache_io(
                artifacts["decode"].artifact, masked_softmax=True
            ),
            kv_cache_io.to_kv_cache_io(
                artifacts["prefill"].artifact, masked_softmax=True
            ),
            kv_cache_io.CALIBRATION_MASK_VALUE,
        )

    def test_greedy_matches_model_r3(self):
        artifacts = self.artifacts.components
        self._check_greedy(
            kv_cache_io.to_kv_cache_io(artifacts["decode"].artifact, r3=True),
            kv_cache_io.to_kv_cache_io(artifacts["prefill"].artifact, r3=True),
            -65535.0,
        )

    def _check_greedy(self, dec, pre, masked):
        layers = dec.num_layers
        prompt = [5, 17, 42, 9, 77, 3, 64]  # two prompt chunks of 4, the second padded
        past_k = [torch.zeros(s) for s in dec.k_shapes]
        past_v = [torch.zeros(s) for s in dec.v_shapes]
        ar, past = pre.ar_len, L - pre.ar_len
        with torch.no_grad():
            start = 0
            while start < len(prompt):
                chunk = prompt[start : start + ar]
                n = len(chunk)
                mask = torch.full((1, 1, ar, L), masked)
                for r in range(ar):
                    mask[..., r, :start] = 0
                    mask[..., r, past : past + min(r, n - 1) + 1] = 0
                out = pre.module(
                    torch.tensor([chunk + [chunk[-1]] * (ar - n)], dtype=torch.int32),
                    mask,
                    torch.tensor([[start + i for i in range(ar)]], dtype=torch.int32),
                    *[k[..., :past] for k in past_k],
                    *[v[..., :past, :] for v in past_v],
                )
                for i in range(layers):
                    past_k[i][..., start : start + n] = out[1 + i][..., :n]
                    past_v[i][..., start : start + n, :] = out[1 + layers + i][
                        ..., :n, :
                    ]
                last = out[0][0, n - 1]
                start += n
            tokens = prompt + [int(last.argmax())]
            for _ in range(8):
                pos = len(tokens) - 1
                mask = torch.full((1, 1, 1, L), masked)
                mask[..., :pos] = 0
                mask[..., L - 1 :] = 0
                out = dec.module(
                    torch.tensor([[tokens[-1]]], dtype=torch.int32),
                    mask,
                    torch.tensor([[pos]], dtype=torch.int32),
                    *past_k,
                    *past_v,
                )
                for i in range(layers):
                    past_k[i][..., pos] = out[1 + i][..., 0]
                    past_v[i][..., pos, :] = out[1 + layers + i][..., 0, :]
                tokens.append(int(out[0][0, -1].argmax()))
        expected = self.model.generate(
            torch.tensor([prompt]), max_new_tokens=9, do_sample=False
        )[0].tolist()
        self.assertEqual(tokens, expected)


class TestKvCacheIoQkNorm(TestKvCacheIo):
    """Qwen3: per-head query and key norms."""

    make_model = staticmethod(_qwen3_model)

    def test_qk_norm_after_head_transpose(self):
        for step in (self.decode, self.prefill):
            means = [
                n
                for n in step.export().graph.nodes
                if n.target == torch.ops.aten.mean.dim and n.meta["val"].dim() == 4
            ]
            self.assertEqual(len(means), 4)  # q and k norms of 2 layers
            for mean in means:
                # Normalized as [B, H, S, D], the layout the attention reads.
                _, heads, steps, _ = mean.meta["val"].shape
                self.assertIn(heads, (2, 4))
                self.assertEqual(steps, step.ar_len)


@unittest.skipIf(DynamoExporter is None, "needs transformers.exporters")
class TestKvCacheIoSeq2Seq(unittest.TestCase):
    """An encoder-decoder (tiny random Whisper): self-attention caches as I/O,
    cross-attention keys and values as fixed inputs, positions from the
    caches' counters."""

    def test_greedy_matches_model(self):
        from executorch.backends.qualcomm.utils.hf_export import _CrossAttentionInputs
        from transformers import WhisperConfig, WhisperForConditionalGeneration

        torch.manual_seed(0)
        config = WhisperConfig(
            vocab_size=64,
            num_mel_bins=8,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_source_positions=16,
            max_target_positions=L,
            decoder_start_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
            suppress_tokens=[],
            begin_suppress_tokens=[],
        )
        model = WhisperForConditionalGeneration(config).eval()
        features = torch.randn(1, 8, 32)
        generation = GenerationConfig(
            do_sample=False,
            cache_implementation="static",
            max_cache_len=L,
            max_new_tokens=8,
            decoder_start_token_id=1,
            eos_token_id=2,
        )
        artifacts = DynamoExporter().export_for_generation(
            model,
            {"input_features": features},
            DynamoConfig(),
            generation_config=generation,
        )
        step = kv_cache_io.to_kv_cache_io(artifacts.components["decode"].artifact)
        self.assertEqual(len(step.cross_k_shapes), 2)
        self.assertEqual(step.constant_methods()["get_n_cross_layers"], 2)
        with torch.no_grad():
            cross = _CrossAttentionInputs(model, 1)(features)
            past_k = [torch.zeros(s) for s in step.k_shapes]
            past_v = [torch.zeros(s) for s in step.v_shapes]
            tokens = [1]
            for pos in range(8):
                mask = torch.full((1, 1, 1, L), -65535.0)
                mask[..., :pos] = 0
                mask[..., L - 1 :] = 0
                out = step.module(
                    torch.tensor([[tokens[-1]]], dtype=torch.int32),
                    mask,
                    torch.tensor([[pos]], dtype=torch.int32),
                    *past_k,
                    *past_v,
                    *cross,
                )
                for i in range(step.num_layers):
                    past_k[i][..., pos] = out[1 + i][..., 0]
                    past_v[i][..., pos, :] = out[1 + step.num_layers + i][..., 0, :]
                tokens.append(int(out[0][0, -1].argmax()))
            expected = [1]
            for _ in range(8):
                logits = model(
                    input_features=features, decoder_input_ids=torch.tensor([expected])
                ).logits
                expected.append(int(logits[0, -1].argmax()))
        self.assertEqual(tokens, expected)


if __name__ == "__main__":
    unittest.main()
