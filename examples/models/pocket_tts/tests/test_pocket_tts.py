# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from executorch.examples.models.phonon.model import mimi_zero_state
from executorch.examples.models.phonon.run_phonon import (
    EagerPrograms,
    ExecuTorchPrograms,
)
from executorch.examples.models.phonon.tests.reference import (
    FlowLMReference,
    MimiReference,
)
from executorch.examples.models.pocket_tts.export_pocket_tts import lower_pocket_tts
from executorch.examples.models.pocket_tts.model import (
    build_programs,
    load_voice,
    PocketConfig,
    PocketTTS,
    TimeVarianceNorm,
)
from executorch.examples.models.pocket_tts.run_pocket_tts import prepare_text
from safetensors.torch import save_file
from torch.nn import functional as F
from torchao.quantization import IntxUnpackedToInt8Tensor


def tiny_config() -> PocketConfig:
    return PocketConfig(
        dim=64,
        num_heads=2,
        num_layers=2,
        ffn_dim=128,
        flow_dim=32,
        flow_depth=2,
        latent_dim=8,
        text_bins=50,
        sample_rate=1600,
        frame_rate=50.0,
        mimi_dim=32,
        n_filters=8,
        ratios=(2, 2),
        mimi_heads=2,
        mimi_layers=2,
        mimi_ffn_dim=64,
        mimi_context=12,
        max_context=64,
        max_prefill=16,
    )


def random_model() -> PocketTTS:
    torch.manual_seed(7)
    model = PocketTTS(tiny_config()).eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.2)
    return model


class PocketTTSTest(unittest.TestCase):
    def test_time_condition_uses_sample_variance(self):
        model = random_model()
        flow = model.flow_lm.flow_net
        expected = []
        for embed, time in zip(flow.time_embed, (0.0, 1.0)):
            angle = time * embed.freqs
            x = embed.mlp[2](
                F.silu(embed.mlp[0](torch.cat([angle.cos(), angle.sin()])))
            )
            variance = ((x - x.mean()) ** 2).sum() / (x.numel() - 1)
            expected.append(x * embed.mlp[3].alpha / torch.sqrt(variance + 1e-5))
        torch.testing.assert_close(flow.lsd_time_condition()[0], sum(expected) / 2)
        norm = TimeVarianceNorm(4)
        x = torch.tensor([[1.0, 2.0, 3.0, 8.0]])
        self.assertFalse(
            torch.allclose(
                norm(x), x / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            )
        )

    def test_checkpoint_is_strict_except_voice_encoder(self):
        model = random_model()
        state = model.state_dict()
        state["mimi.encoder.unused"] = torch.zeros(1)
        state["flow_lm.speaker_proj_weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(state, str(path))
            loaded = PocketTTS.from_checkpoint(path, model.cfg)
            self.assertEqual(loaded.rope_layout, "half_split")
            torch.testing.assert_close(
                loaded.flow_lm.conditioner.embed.weight,
                model.flow_lm.conditioner.embed.weight,
            )
            state["unexpected_decoder_weight"] = torch.zeros(1)
            save_file(state, str(path))
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                PocketTTS.from_checkpoint(path, model.cfg)
            del state["unexpected_decoder_weight"]
            del state["flow_lm.out_eos.weight"]
            save_file(state, str(path))
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                PocketTTS.from_checkpoint(path, model.cfg)

    def test_streaming_programs_match_interleaved_reference(self):
        model = random_model()
        reference = FlowLMReference(model)
        mimi_model = copy.deepcopy(model)
        mimi_model.cfg = replace(model.cfg, mimi_context=model.cfg.mimi_context - 1)
        mimi = MimiReference(mimi_model)
        programs = build_programs(copy.deepcopy(model).to_half_split_rope(), "sdpa")
        pos, previous, state = 0, None, mimi_zero_state(model.cfg)
        prompt = torch.randn(1, 7, model.cfg.dim)
        reference.prompt(prompt)
        programs["flow_lm"](
            prompt, torch.tensor([pos]), torch.zeros(1, model.cfg.latent_dim)
        )
        pos += prompt.shape[1]
        embeds = model.flow_lm.bos_embeds()
        for _ in range(5):
            noise = torch.randn(1, model.cfg.latent_dim)
            latent, eos, embeds = programs["flow_lm"](
                embeds, torch.tensor([pos]), noise
            )
            previous, expected_eos = reference.step(previous, noise)
            torch.testing.assert_close(latent, previous[:, 0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(eos, expected_eos, atol=1e-5, rtol=1e-5)
            pcm, *state = programs["mimi_decode"](latent[:, None], *state)
            torch.testing.assert_close(pcm, mimi.decode(previous), atol=1e-5, rtol=1e-5)
            pos += 1

    def test_xnnpack_program_runs_from_file(self):
        model = random_model().to_half_split_rope()
        reference = EagerPrograms(build_programs(copy.deepcopy(model), "sdpa"), {})
        program, delegation, metadata = lower_pocket_tts(model)
        self.assertGreater(delegation["flow_lm"]["delegated_nodes"], 0)
        self.assertGreater(delegation["mimi_decode"]["delegated_nodes"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pocket_tts.pte"
            with path.open("wb") as output:
                program.write_to_file(output)
            runtime = ExecuTorchPrograms(str(path))
            self.assertEqual(runtime.constant("eos_threshold"), -4.0)
            torch.testing.assert_close(
                runtime.constant("bos_embeds"), metadata["bos_embeds"]
            )
            actual_state, expected_state = mimi_zero_state(model.cfg), mimi_zero_state(
                model.cfg
            )
            pos = 0
            # A second utterance starts at zero after filling the cache.
            for seq in (1, model.cfg.max_prefill, 3, 1, 1, 1):
                if seq == 3:
                    pos = 0
                tokens = torch.arange(seq).reshape(1, seq)
                embeds = runtime("text_embed", (tokens,))[0]
                torch.testing.assert_close(
                    embeds, reference("text_embed", (tokens,))[0]
                )
                inputs = (
                    embeds,
                    torch.tensor([pos]),
                    torch.randn(1, model.cfg.latent_dim),
                )
                actual = runtime("flow_lm", inputs)
                expected = reference("flow_lm", inputs)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
                inputs = (expected[0][:, None],)
                decoded = runtime("mimi_decode", (*inputs, *actual_state))
                eager = reference("mimi_decode", (*inputs, *expected_state))
                for got, want in zip(decoded, eager):
                    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
                actual_state = [value.clone() for value in decoded[1:]]
                expected_state = eager[1:]
                pos += seq

    def test_quantized_xnnpack_prefill_decode_and_reset(self):
        for quantize in ("4bit", "8bit"):
            with self.subTest(
                quantize=quantize
            ), tempfile.TemporaryDirectory() as directory:
                model = random_model().to_half_split_rope()
                reference = build_programs(model, "sdpa")["flow_lm"]
                program, delegation, _ = lower_pocket_tts(model, "xnnpack", quantize)
                self.assertFalse(
                    any(
                        "quant" in op
                        for op in delegation["flow_lm"]["non_delegated_ops"]
                    )
                )
                path = Path(directory) / "pocket_tts.pte"
                with path.open("wb") as output:
                    program.write_to_file(output)
                runtime = ExecuTorchPrograms(str(path))
                if quantize == "4bit":
                    # XNNPACK stores INT4 group scales in BF16.
                    for parameter in reference.parameters():
                        if isinstance(parameter, IntxUnpackedToInt8Tensor):
                            parameter.scale = parameter.scale.bfloat16().float()
                pos = 0
                for seq in (model.cfg.max_prefill, 1, 1, 3, 1):
                    if seq == 3:
                        pos = 0
                    args = (
                        torch.randn(1, seq, model.cfg.dim),
                        torch.tensor([pos]),
                        torch.randn(1, model.cfg.latent_dim),
                    )
                    for actual, expected in zip(
                        runtime("flow_lm", args), reference(*args)
                    ):
                        torch.testing.assert_close(
                            actual, expected, atol=1e-4, rtol=1e-4
                        )
                    pos += seq

    def test_voice_and_text_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.safetensors"
            voice = torch.randn(1, 5, 64)
            save_file({"audio_prompt": voice}, str(path))
            torch.testing.assert_close(load_voice(path, 64), voice)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_voice(path, 32)
            save_file({"speaker_wavs": voice}, str(path))
            with self.assertRaisesRegex(ValueError, "audio_prompt"):
                load_voice(path, 64)
        self.assertEqual(prepare_text(" hello "), ("        Hello.", 5))
        self.assertEqual(
            prepare_text("This is a longer test sentence!"),
            ("This is a longer test sentence!", 3),
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_text("  ")
