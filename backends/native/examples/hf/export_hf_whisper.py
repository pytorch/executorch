# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a Hugging Face Whisper model through the native backend.

Builds ``WhisperForConditionalGeneration`` from a Hugging Face model
directory (or only its config, with random weights) and exports two methods
into one ``.ptn``:

- ``encoder``: ``input_features [1, num_mel_bins, 2 * max_source_positions]``
  (30 s of log-mel at the usual 100 fps) -> ``encoder_hidden_states
  [1, max_source_positions, d_model]``.
- ``cross_kv``: ``encoder_hidden_states`` -> per-layer cross-attention
  keys then values, ``2 * decoder_layers`` tensors of
  ``[1, decoder_attention_heads, max_source_positions, head_dim]``,
  projected and reshaped exactly as ``WhisperAttention`` would on first use.
- ``decoder``: ``(decoder_input_ids [1, T], cache_position [T],
  *cross_keys, *cross_values)`` -> ``logits [1, T, vocab_size]`` with a
  static self-attention KV cache of ``max_target_positions`` rows held in
  mutable buffers, mirroring the decoder-only wrapper in ``export_hf_llm``.
  The runner projects cross K/V once, prefills the forced prefix, then
  decodes one token per call; positions and the causal mask derive from
  ``cache_position`` in-graph, so there is no attention-mask input. The
  Vulkan engine lowers the cached block (learned positions, no RoPE) in
  ``LowerHFAttention`` and cross/encoder SDPA in ``LowerSDPA``. Cross K/V
  reach the decoder as ordinary inputs set once before prefill (input
  staging persists across ``execute()`` calls); zero-copy sharing between
  executables is tracked in backends/native/runtime/vulkan/TODO.md.

Reports which operators the native partitioner leaves behind, like
``export_hf_llm``. ``--dump_dir`` additionally writes the example mel input
and an eager greedy transcript (``input_features.bin``, ``prompt_ids.txt``,
``expected_ids.txt``) for the ``examples/audio`` runner.
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.nn as nn
from executorch.backends.native.examples.hf.hf_utils import report_lowering
from executorch.exir.native import to_native
from torch.export import Dim, export, ExportedProgram
from transformers import (
    AutoConfig,
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from transformers.integrations.executorch import (
    register_dynamic_cache_export_support,
)
from transformers.utils import GENERATION_CONFIG_NAME
from transformers.utils.hub import cached_file

logger: logging.Logger = logging.getLogger(__name__)


def _load(
    path: str,
    encoder_layers: int,
    decoder_layers: int,
    random_weights: bool,
) -> WhisperForConditionalGeneration:
    config = AutoConfig.from_pretrained(path)
    if not isinstance(config, WhisperConfig):
        raise ValueError(f"{path!r} is a {type(config).__name__}, not Whisper.")
    if encoder_layers:
        config.encoder_layers = encoder_layers
    if decoder_layers:
        config.decoder_layers = decoder_layers
    kwargs = {"torch_dtype": torch.float32, "attn_implementation": "sdpa"}
    if random_weights:
        model = WhisperForConditionalGeneration(config)
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
        # The constructor synthesizes a generation config from the model
        # config; prefer the checkpoint's (it carries forced decoder ids).
        generation_config_file = cached_file(
            path,
            GENERATION_CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        if generation_config_file is None:
            logger.warning(
                "No generation config found in %s; using the model config's defaults",
                path,
            )
        else:
            model.generation_config = GenerationConfig.from_pretrained(path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            path, config=config, **kwargs
        )
    return model.eval()


class WhisperEncoder(nn.Module):
    """Log-mel ``input_features`` -> encoder hidden states."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state


class WhisperCrossKV(nn.Module):
    """Project encoder hidden states to per-layer cross-attention K/V.

    Returns keys for every decoder layer followed by values, each
    ``[1, heads, source_positions, head_dim]`` — the tensors
    ``WhisperAttention`` computes from ``encoder_hidden_states`` on first
    use (``k_proj``/``v_proj``, head-major, contiguous; cross K/V are not
    scaled, only queries are).
    """

    def __init__(self, model: WhisperForConditionalGeneration) -> None:
        super().__init__()
        self.layers = model.get_decoder().layers
        config = model.config
        self.num_heads = config.decoder_attention_heads
        self.head_dim = config.d_model // config.decoder_attention_heads
        self.source_positions = config.max_source_positions

    def forward(
        self, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # Compute in return order (all keys, then all values): the native
        # partitioner emits delegate outputs in computation order, and
        # to_native requires the method to return them in that order.
        shape = (1, self.source_positions, self.num_heads, self.head_dim)
        keys = [
            layer.encoder_attn.k_proj(encoder_hidden_states)
            .view(*shape)
            .transpose(1, 2)
            .contiguous()
            for layer in self.layers
        ]
        values = [
            layer.encoder_attn.v_proj(encoder_hidden_states)
            .view(*shape)
            .transpose(1, 2)
            .contiguous()
            for layer in self.layers
        ]
        return (*keys, *values)


class WhisperDecoder(nn.Module):
    """Static-cache decoder pass over precomputed cross K/V -> logits.

    Self-attention K/V live in a ``StaticCache`` registered as mutable
    buffers (zero-initialized, as the Vulkan cache rewrite requires). The
    cross-attention cache is marked updated and its layers are bound to
    the ``cross_kv`` inputs (keys for every layer, then values) on each
    call, so ``WhisperAttention`` reads them directly and never projects
    encoder states. The cache is stateful: one instance decodes one
    utterance.
    """

    def __init__(self, model: WhisperForConditionalGeneration) -> None:
        super().__init__()
        self.decoder = model.get_decoder()
        self.proj_out = model.proj_out
        config = model.config
        head_dim = config.d_model // config.decoder_attention_heads
        self.self_cache = StaticCache(
            config=config, max_cache_len=config.max_target_positions
        )
        self.self_cache.early_initialization(
            batch_size=1,
            num_heads=config.decoder_attention_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.cross_cache = DynamicCache(config=config)
        self.cache = EncoderDecoderCache(self.self_cache, self.cross_cache)
        for layer_idx in range(config.decoder_layers):
            self.cache.is_updated[layer_idx] = True
        register_dynamic_cache_export_support()
        for i in range(len(self.self_cache)):
            self.register_buffer(
                f"key_cache_{i}", self.self_cache.layers[i].keys, persistent=False
            )
            self.register_buffer(
                f"value_cache_{i}",
                self.self_cache.layers[i].values,
                persistent=False,
            )

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        *cross_kv: torch.Tensor,
    ) -> torch.Tensor:
        num_layers = len(self.cross_cache.layers)
        for layer_idx, layer in enumerate(self.cross_cache.layers):
            layer.keys = cross_kv[layer_idx]
            layer.values = cross_kv[num_layers + layer_idx]
            layer.is_initialized = True
        hidden = self.decoder(
            input_ids=decoder_input_ids,
            # Gate only: WhisperDecoderLayer runs its cross-attention block
            # iff this is not None, but with the cross cache marked updated
            # WhisperAttention reads K/V from the cache and never touches
            # these values. Any input tensor serves; reuse cross keys 0.
            encoder_hidden_states=cross_kv[0],
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        ).last_hidden_state
        return self.proj_out(hidden)


def prompt_ids(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor | None = None,
) -> list[int]:
    """Forced decoder prefix, detecting an unspecified language from the input."""
    ids = [model.config.decoder_start_token_id]
    forced = getattr(model.generation_config, "forced_decoder_ids", None) or []
    for position, token in sorted(forced):
        if position != len(ids):
            raise ValueError("forced_decoder_ids must form a contiguous prefix")
        if token is None:
            if position != 1:
                raise ValueError("only the language token at position 1 may be None")
            token = int(
                model.detect_language(
                    input_features=input_features,
                    num_segment_frames=2 * model.config.max_source_positions,
                ).item()
            )
        ids.append(token)
    return ids


def example_input_features(config: WhisperConfig) -> torch.Tensor:
    """Seeded stand-in mel input; the feature extractor pads audio to this shape."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(
        (1, config.num_mel_bins, 2 * config.max_source_positions),
        generator=generator,
    )


def build_exported_programs(
    model: WhisperForConditionalGeneration,
    example_features: torch.Tensor | None = None,
) -> tuple[WhisperEncoder, WhisperCrossKV, WhisperDecoder, dict[str, ExportedProgram]]:
    """Export the encoder/cross-KV/decoder wrappers around ``model``.

    The encoder and cross-KV shapes are static (Whisper always consumes a
    full 30 s mel window); only the decoder token axis is dynamic.
    """
    config = model.config
    encoder = WhisperEncoder(model.get_encoder()).eval()
    cross_kv = WhisperCrossKV(model).eval()
    decoder = WhisperDecoder(model).eval()
    if example_features is None:
        example_features = example_input_features(config)
    batch = example_features.shape[0]

    with torch.no_grad():
        hidden = encoder(example_features)
        cross = cross_kv(hidden)
        # Trace with a handful of tokens: a 1-token example would specialize
        # the dynamic token axis to a constant.
        example_ids = prompt_ids(model, example_features)
        example_ids += [example_ids[-1]] * max(0, 4 - len(example_ids))
        prefix = torch.tensor([example_ids], dtype=torch.long)
        prefix = prefix.expand(batch, -1).contiguous()
        cache_position = torch.arange(len(example_ids), dtype=torch.long)
        tokens = Dim("tokens", min=1, max=config.max_target_positions)
        encoder_ep = export(
            encoder, (example_features,), dynamic_shapes=None, strict=True
        )
        cross_kv_ep = export(cross_kv, (hidden,), dynamic_shapes=None, strict=True)
        decoder_ep = export(
            decoder,
            (prefix, cache_position, *cross),
            dynamic_shapes=(
                {1: tokens},
                {0: tokens},
                (None,) * len(cross),
            ),
            strict=True,
        )
    programs = {
        "encoder": encoder_ep,
        "cross_kv": cross_kv_ep,
        "decoder": decoder_ep,
    }
    return encoder, cross_kv, decoder, programs


def _greedy_transcript(
    encoder: WhisperEncoder,
    cross_kv: WhisperCrossKV,
    decoder: WhisperDecoder,
    example_features: torch.Tensor,
    prefix: list[int],
    eos_ids: set[int],
    max_new_tokens: int,
) -> list[int]:
    ids = list(prefix)
    capacity = decoder.decoder.config.max_target_positions
    if not ids or len(ids) > capacity:
        raise ValueError(
            "the prefix must contain between 1 and max_target_positions tokens"
        )
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    max_new_tokens = min(max_new_tokens, capacity - len(ids))
    if max_new_tokens == 0:
        return ids
    with torch.no_grad():
        cross = cross_kv(encoder(example_features))
        window = torch.tensor([ids], dtype=torch.long)
        positions = torch.arange(len(ids), dtype=torch.long)
        logits = decoder(window, positions, *cross)
        for step in range(max_new_tokens):
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            if next_id in eos_ids or step + 1 == max_new_tokens:
                break
            logits = decoder(
                torch.tensor([[next_id]], dtype=torch.long),
                torch.tensor([len(ids) - 1], dtype=torch.long),
                *cross,
            )
    return ids


def _dump_test_vectors(
    dump_dir: str,
    model: WhisperForConditionalGeneration,
    encoder: WhisperEncoder,
    example_features: torch.Tensor,
    max_new_tokens: int,
) -> None:
    os.makedirs(dump_dir, exist_ok=True)
    # A fresh decoder: the exported one's cache buffers must stay zero for
    # the serialized package, and eager decoding would fill them.
    decoder = WhisperDecoder(model).eval()
    prefix = prompt_ids(model, example_features)
    eos = model.config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    transcript = _greedy_transcript(
        encoder,
        WhisperCrossKV(model).eval(),
        decoder,
        example_features,
        prefix,
        eos_ids,
        max_new_tokens,
    )
    with open(os.path.join(dump_dir, "input_features.bin"), "wb") as f:
        f.write(example_features.contiguous().numpy().tobytes())
    for name, ids in (("prompt_ids.txt", prefix), ("expected_ids.txt", transcript)):
        with open(os.path.join(dump_dir, name), "w") as f:
            f.write("".join(f"{token}\n" for token in ids))
    print(f"prompt ids: {prefix}")
    print(f"eos ids: {sorted(eos_ids)}")
    print(f"eager greedy ids ({len(transcript)}): {transcript}")
    print(f"wrote test vectors to {dump_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="Hugging Face model directory or hub ID.",
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Use only the model's config, with random weights.",
    )
    parser.add_argument("--encoder_layers", type=int, default=0)
    parser.add_argument("--decoder_layers", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--dump_dir",
        default="",
        help="Also write runner test vectors (mel input, prompt, eager ids).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum new test-vector tokens, capped by the decoder context.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    model = _load(
        args.model, args.encoder_layers, args.decoder_layers, args.random_weights
    )
    example_features = example_input_features(model.config)
    encoder, _cross_kv, _decoder, programs = build_exported_programs(
        model, example_features
    )
    for name, ep in programs.items():
        report_lowering(ep, name)

    if args.dump_dir:
        _dump_test_vectors(
            args.dump_dir,
            model,
            encoder,
            example_features,
            args.max_new_tokens,
        )
    if args.output:
        to_native(programs).save(args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
