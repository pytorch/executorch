# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthesize speech with an exported Phonon program (ExecuTorch pybindings).

  python -m executorch.examples.models.phonon.run_phonon \\
      --model phonon_exports/xnnpack/phonon.pte \\
      --text "Hello from ExecuTorch." --output hello.wav

The host side is small and backend independent: basic text preparation and
sentence chunking, tokenization with tokenizer.json, the noise of the flow
head, the end-of-speech rule, and feeding next_embeds and the Mimi state back.
"""

import argparse
import math
import re
import time
import wave
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

MAX_CHUNK_TOKENS = 50  # the official runtime's sentence chunk size


def prepare_text(text: str) -> Tuple[str, int]:
    """Basic text preparation of the official runtime: collapse whitespace,
    capitalize the first letter, end with a period if the text ends with a
    letter or digit. Returns the text and how many frames to keep generating
    after the end-of-speech logit fires (3 for texts of up to 4 words, else 1).

    Letters and digits are ASCII, as in the C++ runner. Numbers, symbols,
    abbreviations, URLs and other non-word input are not normalized (the
    official runtime has English rules for them)."""
    words = text.split()
    text = " ".join(words)
    if not text:
        return text, 3
    frames_after_eos = 3 if len(words) <= 4 else 1
    if "a" <= text[0] <= "z":
        text = text[0].upper() + text[1:]
    if text[-1].isascii() and text[-1].isalnum():
        text += "."
    return text, frames_after_eos


def split_sentences(text: str, count_tokens: Callable[[str], int], max_tokens: int):
    """Greedily pack whole sentences into chunks of at most max_tokens tokens;
    a longer sentence is split at word boundaries."""
    chunks: List[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        pieces = [sentence]
        if count_tokens(sentence) > max_tokens:
            pieces = []
            piece = ""
            for word in sentence.split():
                candidate = f"{piece} {word}".strip()
                if piece and count_tokens(candidate) > max_tokens:
                    pieces.append(piece)
                    candidate = word
                piece = candidate
            pieces.append(piece)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and count_tokens(candidate) > max_tokens:
                chunks.append(current)
                candidate = piece
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def frame_budget(num_tokens: int, frame_rate: float) -> int:
    return math.ceil((num_tokens / 3.0 + 2.0) * frame_rate)


class NoiseGenerator:
    """Gaussian noise of the flow head, identical to phonon::NoiseGenerator in
    the C++ runner: splitmix64 uniforms and the Box-Muller transform."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def _uniform(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        z ^= z >> 31
        return (z >> 11) * 2.0**-53

    def normal(self, count: int, stddev: float) -> torch.Tensor:
        out = []
        while len(out) < count:
            radius = math.sqrt(-2.0 * math.log(1.0 - self._uniform()))
            angle = 2.0 * math.pi * self._uniform()
            out += [
                radius * math.cos(angle) * stddev,
                radius * math.sin(angle) * stddev,
            ]
        return torch.tensor(out[:count], dtype=torch.float32)


class ExecuTorchPrograms:
    """Methods of an exported phonon.pte, run with the ExecuTorch pybindings."""

    def __init__(self, model_path: str, data_path: Optional[str] = None):
        # Kernels for llama::sdpa_with_kv_cache (CPU backends) and quantized ops.
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        try:
            import executorch.kernels.quantized  # noqa: F401
        except ImportError:
            pass
        self.module = _load_for_executorch(model_path, data_path=data_path)

    def __call__(self, method: str, inputs: Sequence[torch.Tensor]) -> List:
        return list(self.module.run_method(method, [x.contiguous() for x in inputs]))

    def constant(self, name: str):
        return self.module.run_method(name, [])[0]

    def mimi_zero_state(self) -> List[torch.Tensor]:
        meta = self.module.method_meta("mimi_decode")
        return [
            torch.zeros(tuple(meta.input_tensor_meta(i).sizes()))
            for i in range(1, meta.num_inputs())
        ]


class EagerPrograms:
    """The same methods as eager PyTorch modules (for parity checks)."""

    def __init__(self, programs: dict, metadata: dict):
        self.programs, self.metadata = programs, metadata

    def __call__(self, method: str, inputs: Sequence[torch.Tensor]) -> List:
        with torch.no_grad():
            out = self.programs[method](*inputs)
        return [t.clone() for t in (out if isinstance(out, tuple) else (out,))]

    def constant(self, name: str):
        return self.metadata[name]

    def mimi_zero_state(self) -> List[torch.Tensor]:
        from executorch.examples.models.phonon.model import mimi_zero_state

        return mimi_zero_state(self.programs["mimi_decode"].cfg)


def load_tokenizer(path: str | Path) -> Callable[[str], List[int]]:
    """Encoder for the upstream tokenizer.json (no special tokens)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


class PhononRunner:
    def __init__(self, programs, tokenize: Callable[[str], List[int]]):
        self.run = programs
        self.tokenize = tokenize
        c = programs.constant
        if c("format_version") != 2:
            raise ValueError("Unsupported Phonon program; re-export it")
        self.mimi_state_in_program = bool(c("mimi_state_in_program"))
        self.sample_rate = c("sample_rate")
        self.frame_samples = c("frame_samples")
        self.frame_rate = c("frame_rate")
        self.latent_dim = c("latent_dim")
        self.max_context = c("max_context")
        self.max_prefill = c("max_prefill")
        self.eos_threshold = c("eos_threshold")
        self.default_temperature = c("temperature")
        self.bos_embeds = c("bos_embeds")

    def _flow(self, embeds, pos, noise):
        latent, eos, next_embeds = self.run(
            "flow_lm", [embeds, torch.tensor([pos], dtype=torch.long), noise]
        )
        return latent, float(eos.reshape(-1)[0]), next_embeds

    def synthesize(
        self,
        text: str,
        voice: torch.Tensor,
        temperature: Optional[float] = None,
        seed: int = 0,
        noise_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Returns float PCM at sample_rate and generation info. noise_fn(k)
        overrides the noise of step k (default: N(0, temperature) from
        NoiseGenerator(seed))."""
        temperature = self.default_temperature if temperature is None else temperature
        if noise_fn is None:
            gen, scale = NoiseGenerator(seed), math.sqrt(temperature)

            def noise_fn(k):
                return gen.normal(self.latent_dim, scale).reshape(1, -1)

        voice_embeds = self.run("voice_embed", [voice])[0]
        voice_len = voice_embeds.shape[1]
        if voice_len > self.max_prefill:
            raise ValueError("Voice prompt is longer than max_prefill")
        self._flow(voice_embeds, 0, torch.zeros(1, self.latent_dim))
        # The inputs of mimi_decode after the latent: the state, or reset = 1
        # then 0 when the program keeps the state.
        stateful = self.mimi_state_in_program
        mimi_state = [torch.ones(1)] if stateful else self.run.mimi_zero_state()
        prepared, _ = prepare_text(text)
        chunks = split_sentences(
            prepared, lambda s: len(self.tokenize(s)), MAX_CHUNK_TOKENS
        )
        pcm, info, step = [], {"chunks": [], "latents": [], "eos_logits": []}, 0
        for chunk in chunks:
            chunk, frames_after_eos = prepare_text(chunk)
            tokens = self.tokenize(chunk)
            if not 0 < len(tokens) <= self.max_prefill:
                raise ValueError(f"Chunk has an unsupported token count: {chunk}")
            pos = voice_len
            text_embeds = self.run("text_embed", [torch.tensor([tokens])])[0]
            self._flow(text_embeds, pos, torch.zeros(1, self.latent_dim))
            pos += len(tokens)
            budget = min(
                frame_budget(len(tokens), self.frame_rate), self.max_context - pos
            )
            embeds, countdown, frames = self.bos_embeds, None, 0
            for _ in range(budget):
                latent, eos_logit, embeds = self._flow(embeds, pos, noise_fn(step))
                pos, step, frames = pos + 1, step + 1, frames + 1
                out = self.run("mimi_decode", [latent.reshape(1, 1, -1), *mimi_state])
                pcm.append(out[0].reshape(-1).numpy().copy())
                mimi_state = [torch.zeros(1)] if stateful else out[1:]
                info["latents"].append(latent.reshape(-1).numpy().copy())
                info["eos_logits"].append(eos_logit)
                if eos_logit > self.eos_threshold and countdown is None:
                    countdown = frames_after_eos
                if countdown is not None:
                    if countdown == 0:
                        break
                    countdown -= 1
            info["chunks"].append({"text": chunk, "tokens": tokens, "frames": frames})
        info["frames"] = step
        return np.concatenate(pcm) if pcm else np.zeros(0, np.float32), info


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    pcm16 = np.clip(np.round(pcm.astype(np.float64) * 32767), -32768, 32767)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="phonon.pte")
    parser.add_argument("--data-path", help="Data file (aoti_cuda_blob.ptd for CUDA)")
    parser.add_argument(
        "--tokenizer", help="tokenizer.json (default: next to the model)"
    )
    parser.add_argument(
        "--voice",
        help="Voice .safetensors with speaker_wavs (default: default-voice.safetensors "
        "next to the model)",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="phonon.wav")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Flow noise variance (default 0.3; 0 removes the noise)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from executorch.examples.models.phonon.model import load_voice

    model_dir = Path(args.model).parent
    t0 = time.perf_counter()
    runner = PhononRunner(
        ExecuTorchPrograms(args.model, args.data_path),
        load_tokenizer(args.tokenizer or model_dir / "tokenizer.json"),
    )
    load_s = time.perf_counter() - t0
    voice = load_voice(args.voice or model_dir / "default-voice.safetensors")
    t0 = time.perf_counter()
    pcm, info = runner.synthesize(args.text, voice, args.temperature, args.seed)
    synth_s = time.perf_counter() - t0
    write_wav(args.output, pcm, runner.sample_rate)
    audio_s = len(pcm) / runner.sample_rate
    print(
        f"Wrote {args.output}: {audio_s:.2f} s of audio, {info['frames']} frames, "
        f"load {load_s:.2f} s, synthesis {synth_s:.2f} s (RTF {synth_s / max(audio_s, 1e-9):.2f})"
    )


if __name__ == "__main__":
    main()
