#!/usr/bin/env python3
"""MLX helpers for Qwen3-TTS benchmarking and session reuse."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


MLX_AUDIO_REPO_ENV = "MLX_AUDIO_REPO"


def _purge_mlx_audio_modules() -> None:
    for name in list(sys.modules):
        if name == "mlx_audio" or name.startswith("mlx_audio."):
            sys.modules.pop(name, None)


def _resolve_mlx_audio_repo(explicit_repo: Optional[Path]) -> Optional[Path]:
    candidates = []
    if explicit_repo is not None:
        candidates.append(Path(explicit_repo).expanduser().resolve())
    env_repo = os.environ.get(MLX_AUDIO_REPO_ENV)
    if env_repo:
        candidates.append(Path(env_repo).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[3]
    candidates.append((repo_root.parent / "mlx-audio").resolve())

    for candidate in candidates:
        if (candidate / "mlx_audio").exists():
            return candidate
    return None


def _load_mlx_symbols(explicit_repo: Optional[Path]):
    repo = _resolve_mlx_audio_repo(explicit_repo)
    if repo is not None:
        repo_str = str(repo)
        if not sys.path or sys.path[0] != repo_str:
            sys.path.insert(0, repo_str)
        _purge_mlx_audio_modules()

    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio

    return repo, mx, load_model, load_audio


@dataclass
class PromptBenchmark:
    elapsed_s: float
    audio_s: float
    throughput_x: float
    first_audio_s: float
    chunk_count: int
    sample_count: int


def _collect_prompt_benchmark(mx, results, elapsed_s: float) -> PromptBenchmark:
    if not results:
        raise RuntimeError("MLX generate returned no results.")

    audio_chunks = [result.audio for result in results]
    if len(audio_chunks) == 1:
        waveform = audio_chunks[0]
    else:
        waveform = mx.concatenate(audio_chunks, axis=0)
    mx.eval(waveform)

    sample_rate = getattr(results[-1], "sample_rate", 24000)
    sample_count = int(waveform.shape[-1])
    audio_s = sample_count / sample_rate if sample_rate > 0 else 0.0
    throughput_x = audio_s / elapsed_s if elapsed_s > 0.0 else 0.0
    if len(results) == 1:
        first_audio_s = elapsed_s
    else:
        first_audio_s = getattr(results[0], "processing_time_seconds", elapsed_s)

    return PromptBenchmark(
        elapsed_s=elapsed_s,
        audio_s=audio_s,
        throughput_x=throughput_x,
        first_audio_s=first_audio_s,
        chunk_count=len(results),
        sample_count=sample_count,
    )


class Qwen3TTSMlxBackend:
    """Loads the local mlx-audio Qwen3-TTS model once and reuses it."""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        mlx_audio_repo: Optional[Path] = None,
    ) -> None:
        repo, mx, load_model, load_audio = _load_mlx_symbols(mlx_audio_repo)
        self.repo_path = repo
        self.mx = mx
        self._load_audio = load_audio
        self.model_path = model_path
        self.model = load_model(model_path)

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    def warmup(
        self,
        *,
        text: str,
        ref_audio,
        ref_text: str,
        stream: bool,
        streaming_interval: float = 4.0,
        seed: Optional[int] = None,
    ) -> PromptBenchmark:
        return self.benchmark_baseline(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            stream=stream,
            streaming_interval=streaming_interval,
            seed=seed,
        )

    def benchmark_baseline(
        self,
        *,
        text: str,
        ref_audio,
        ref_text: str,
        stream: bool,
        streaming_interval: float = 4.0,
        seed: Optional[int] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        max_tokens: int = 4096,
    ) -> PromptBenchmark:
        if isinstance(ref_audio, Path):
            ref_audio = str(ref_audio)
        if seed is not None:
            self.mx.random.seed(seed)
        started_at = time.perf_counter()
        results = list(
            self.model.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                stream=stream,
                streaming_interval=streaming_interval,
            )
        )
        elapsed_s = time.perf_counter() - started_at
        return _collect_prompt_benchmark(self.mx, results, elapsed_s)

    def create_icl_session(
        self,
        *,
        ref_audio,
        ref_text: str,
        language: str = "auto",
    ) -> "Qwen3TTSMlxIclSession":
        return Qwen3TTSMlxIclSession(
            backend=self,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
        )


class Qwen3TTSMlxIclSession:
    """Caches the reference ICL conditioning across prompts."""

    def __init__(
        self,
        *,
        backend: Qwen3TTSMlxBackend,
        ref_audio,
        ref_text: str,
        language: str = "auto",
    ) -> None:
        self.backend = backend
        self.mx = backend.mx
        self.model = backend.model
        self.ref_text = ref_text
        self.language = language
        if isinstance(ref_audio, Path):
            ref_audio = str(ref_audio)
        if isinstance(ref_audio, str):
            self.ref_audio = backend._load_audio(ref_audio, sample_rate=self.model.sample_rate)
        else:
            self.ref_audio = ref_audio
        self._cached = self._build_cached_icl_state()

    def _build_cached_icl_state(self):
        if self.model.tokenizer is None:
            raise ValueError("Tokenizer not loaded on the MLX model.")
        if self.model.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded on the MLX model.")

        config = self.model.config.talker_config
        ref_audio = self.ref_audio
        audio_for_spk = ref_audio
        if ref_audio.ndim == 1:
            ref_audio = ref_audio[None, None, :]
        elif ref_audio.ndim == 2:
            ref_audio = ref_audio[None, :]

        ref_codes = self.model.speech_tokenizer.encode(ref_audio)
        ref_chat = f"<|im_start|>assistant\n{self.ref_text}<|im_end|>\n"
        ref_ids = self.mx.array(self.model.tokenizer.encode(ref_chat))[None, :]
        ref_text_ids = ref_ids[:, 3:-2]

        tts_tokens = self.mx.array(
            [[
                self.model.config.tts_bos_token_id,
                self.model.config.tts_eos_token_id,
                self.model.config.tts_pad_token_id,
            ]]
        )
        tts_embeds = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        ref_text_embed = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(ref_text_ids)
        )
        role_ids = self.mx.array(self.model.tokenizer.encode("<|im_start|>assistant\n"))[
            None, :
        ]
        role_embed = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(role_ids)
        )

        first_cb_codes = ref_codes[:, 0, :]
        ref_codec_embed = self.model.talker.get_input_embeddings()(first_cb_codes)
        for i in range(config.num_code_groups - 1):
            cb_codes = ref_codes[:, i + 1, :]
            ref_codec_embed = (
                ref_codec_embed
                + self.model.talker.code_predictor.codec_embedding[i](cb_codes)
            )

        codec_bos_embed = self.model.talker.get_input_embeddings()(
            self.mx.array([[config.codec_bos_id]])
        )
        codec_embed_icl = self.mx.concatenate(
            [codec_bos_embed, ref_codec_embed],
            axis=1,
        )
        codec_pad_embed = self.model.talker.get_input_embeddings()(
            self.mx.array([[config.codec_pad_id]])
        )
        codec_with_text_pad = codec_embed_icl + self.mx.broadcast_to(
            tts_pad_embed, (1, codec_embed_icl.shape[1], tts_pad_embed.shape[-1])
        )
        ref_text_with_codec_pad = ref_text_embed + self.mx.broadcast_to(
            codec_pad_embed, (1, ref_text_embed.shape[1], codec_pad_embed.shape[-1])
        )
        eos_with_codec_pad = tts_eos_embed + codec_pad_embed

        language_id = None
        if self.language.lower() != "auto" and config.codec_language_id:
            language_id = config.codec_language_id.get(self.language.lower())

        speaker_embed = None
        if self.model.speaker_encoder is not None:
            speaker_embed = self.model.extract_speaker_embedding(audio_for_spk)

        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_prefix_embed = self.model.talker.get_input_embeddings()(
            self.mx.array([codec_prefill])
        )
        codec_prefix_suffix = self.model.talker.get_input_embeddings()(
            self.mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )
        if speaker_embed is not None:
            codec_prefix_embed = self.mx.concatenate(
                [
                    codec_prefix_embed,
                    speaker_embed.reshape(1, 1, -1),
                    codec_prefix_suffix,
                ],
                axis=1,
            )
        else:
            codec_prefix_embed = self.mx.concatenate(
                [codec_prefix_embed, codec_prefix_suffix],
                axis=1,
            )

        pad_count = codec_prefix_embed.shape[1] - 2
        pad_embeds = self.mx.broadcast_to(
            tts_pad_embed,
            (1, pad_count, tts_pad_embed.shape[-1]),
        )
        combined_prefix = self.mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]

        self.mx.eval(
            ref_codes,
            ref_text_embed,
            role_embed,
            tts_eos_embed,
            tts_pad_embed,
            codec_pad_embed,
            codec_with_text_pad,
            ref_text_with_codec_pad,
            eos_with_codec_pad,
            combined_prefix,
        )

        return SimpleNamespace(
            ref_codes=ref_codes,
            ref_text_embed=ref_text_embed,
            role_embed=role_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            codec_pad_embed=codec_pad_embed,
            codec_with_text_pad=codec_with_text_pad,
            ref_text_with_codec_pad=ref_text_with_codec_pad,
            eos_with_codec_pad=eos_with_codec_pad,
            combined_prefix=combined_prefix,
        )

    def _prepare_cached_icl_generation_inputs(
        self,
        text: str,
        ref_audio,
        ref_text: str,
        language: str = "auto",
    ):
        del ref_audio, ref_text
        if language.lower() != self.language.lower():
            raise ValueError(
                "Cached ICL session language mismatch: "
                f"expected {self.language!r}, got {language!r}."
            )

        target_chat = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        target_ids = self.mx.array(self.model.tokenizer.encode(target_chat))[None, :]
        text_ids = target_ids[:, 3:-5]

        target_text_embed = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(text_ids)
        )
        target_text_with_codec_pad = target_text_embed + self.mx.broadcast_to(
            self._cached.codec_pad_embed,
            (1, target_text_embed.shape[1], self._cached.codec_pad_embed.shape[-1]),
        )
        text_with_codec_pad = self.mx.concatenate(
            [
                self._cached.ref_text_with_codec_pad,
                target_text_with_codec_pad,
                self._cached.eos_with_codec_pad,
            ],
            axis=1,
        )
        icl_input_embed = self.mx.concatenate(
            [text_with_codec_pad, self._cached.codec_with_text_pad],
            axis=1,
        )
        input_embeds = self.mx.concatenate(
            [self._cached.role_embed, self._cached.combined_prefix, icl_input_embed],
            axis=1,
        )
        return (
            input_embeds,
            self._cached.tts_pad_embed,
            self._cached.tts_pad_embed,
            self._cached.ref_codes,
        )

    def generate(
        self,
        *,
        text: str,
        stream: bool,
        streaming_interval: float = 4.0,
        streaming_context_size: int = 25,
        seed: Optional[int] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        if seed is not None:
            self.mx.random.seed(seed)

        original_prepare = self.model._prepare_icl_generation_inputs

        def iterator():
            self.model._prepare_icl_generation_inputs = (
                self._prepare_cached_icl_generation_inputs
            )
            try:
                yield from self.model._generate_icl(
                    text=text,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    language=self.language,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    verbose=verbose,
                    stream=stream,
                    streaming_interval=streaming_interval,
                    streaming_context_size=streaming_context_size,
                )
            finally:
                self.model._prepare_icl_generation_inputs = original_prepare

        return iterator()

    def benchmark(
        self,
        *,
        text: str,
        stream: bool,
        streaming_interval: float = 4.0,
        streaming_context_size: int = 25,
        seed: Optional[int] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        max_tokens: int = 4096,
    ) -> PromptBenchmark:
        started_at = time.perf_counter()
        results = list(
            self.generate(
                text=text,
                stream=stream,
                streaming_interval=streaming_interval,
                streaming_context_size=streaming_context_size,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            )
        )
        elapsed_s = time.perf_counter() - started_at
        return _collect_prompt_benchmark(self.mx, results, elapsed_s)
