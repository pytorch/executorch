# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthesize a short utterance with an exported Pocket TTS program."""

import argparse
import logging
import math
import wave
from pathlib import Path

import sentencepiece
import torch
from executorch.examples.models.phonon.run_phonon import ExecuTorchPrograms
from executorch.examples.models.pocket_tts.model import load_voice

logger: logging.Logger = logging.getLogger(__name__)


def prepare_text(text: str) -> tuple[str, int]:
    text = text.strip()
    if not text:
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    short = len(text.split()) <= 4
    text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    return (" " * 8 + text if short else text), (5 if short else 3)


class PocketRunner:
    def __init__(self, programs):
        self.run = programs
        self.max_prefill = programs.constant("max_prefill")
        self.max_context = programs.constant("max_context")
        self.latent_dim = programs.constant("latent_dim")

    def prefill(self, embeds: torch.Tensor, pos: int) -> int:
        for chunk in embeds.split(self.max_prefill, dim=1):
            self.run(
                "flow_lm",
                (
                    chunk,
                    torch.tensor([pos]),
                    torch.zeros(1, self.latent_dim),
                ),
            )
            pos += chunk.shape[1]
        return pos

    @torch.no_grad()
    def synthesize(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        *,
        temperature: float = 0.7,
        seed: int = 0,
        frames_after_eos: int = 3,
    ) -> torch.Tensor:
        if not tokens:
            raise ValueError("Text must contain at least one token")
        if not math.isfinite(temperature) or temperature < 0 or frames_after_eos < 1:
            raise ValueError(
                "Temperature must be finite and nonnegative; tail must be positive"
            )
        budget = math.ceil((len(tokens) / 3.0 + 2.0) * self.run.constant("frame_rate"))
        if voice.shape[1] + len(tokens) + budget > self.max_context:
            raise ValueError("Utterance exceeds the KV cache; synthesize shorter text")
        pos = self.prefill(voice, 0)
        for start in range(0, len(tokens), self.max_prefill):
            embeds = self.run(
                "text_embed",
                (torch.tensor([tokens[start : start + self.max_prefill]]),),
            )[0]
            pos = self.prefill(embeds, pos)
        embeds = self.run.constant("bos_embeds").clone()
        state, frames, eos_step = self.run.mimi_zero_state(), [], None
        generator = torch.Generator().manual_seed(seed)
        for step in range(budget):
            noise = torch.randn(1, self.latent_dim, generator=generator) * math.sqrt(
                temperature
            )
            latent, eos, embeds = self.run(
                "flow_lm", (embeds, torch.tensor([pos]), noise)
            )
            embeds = embeds.clone()
            pos += 1
            if eos.item() > self.run.constant("eos_threshold") and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            decoded = self.run("mimi_decode", (latent.reshape(1, 1, -1), *state))
            frames.append(decoded[0].clone().flatten())
            state = [tensor.clone() for tensor in decoded[1:]]
        else:
            logger.warning("Reached generation budget before completing the EOS tail")
        pcm = torch.cat(frames)
        if not torch.isfinite(pcm).all():
            raise RuntimeError("Generated non-finite audio")
        return pcm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("pocket.wav"))
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    programs = ExecuTorchPrograms(str(args.model))
    tokenizer = sentencepiece.SentencePieceProcessor(
        str(args.model.parent / "tokenizer.model")
    )
    text, tail = prepare_text(args.text)
    voice = load_voice(
        args.voice or args.model.parent / "alba.safetensors", programs.constant("dim")
    )
    pcm = PocketRunner(programs).synthesize(
        tokenizer.encode(text),
        voice,
        temperature=args.temperature,
        seed=args.seed,
        frames_after_eos=tail,
    )
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(programs.constant("sample_rate"))
        output.writeframes(
            (pcm.clamp(-1, 1) * 32767).round().short().numpy().astype("<i2").tobytes()
        )
    logger.info(
        "Wrote %s: %.2f seconds",
        args.output,
        pcm.numel() / programs.constant("sample_rate"),
    )


if __name__ == "__main__":
    main()
