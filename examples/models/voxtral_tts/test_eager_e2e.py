"""End-to-end eager FP32 validation for Voxtral TTS.

Loads the model in FP32 eager mode (no export, no quantization) and runs
the full LLM -> flow-matching -> codec pipeline to produce a WAV file.
This serves as the ground truth: if this script produces clear speech,
the architecture is correct and remaining issues are in export/runner.

Matches the reference voxtral-tts.c flow:
  1. Construct prompt embeddings with voice splice
  2. Prefill LLM decoder
  3. Feed AUDIO(24) seed token to get first hidden state
  4. Autoregressive loop: semantic_head -> flow_matching -> audio_embed -> decode
  5. Codec decode -> WAV

Usage:
    python -u test_eager_e2e.py \
        --model-path ~/models/Voxtral-4B-TTS-2603 \
        --text "Hello, how are you today?" \
        --output /tmp/voxtral_eager.wav \
        --max-frames 80
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import torch

from model import (
    END_AUDIO_ID,
    EMPTY_AUDIO_ID,
    N_SPECIAL_TOKENS,
    VoxtralTTSConfig,
    load_model,
    SDPA,
    KVCache,
)
from parity import (
    build_reference_prompt_ids,
    encode_speech_request_tokens,
    run_seed_decode,
    splice_voice_embeddings,
    topk_pairs,
)
from voice import load_voice_from_model_dir


def _patch_eager_sdpa(model):
    """Replace custom_sdpa with standard F.scaled_dot_product_attention.

    The custom_sdpa op is designed for ExecuTorch export and may not behave
    correctly in eager CPU mode.  This monkey-patches every LMAttention layer
    to use PyTorch-native SDPA for ground-truth validation.
    """
    import torch.nn.functional as F

    class EagerKVCache(torch.nn.Module):
        def __init__(self, max_seq_len, n_kv_heads, head_dim):
            super().__init__()
            cache_shape = (1, max_seq_len, n_kv_heads, head_dim)
            self.register_buffer("k_cache", torch.zeros(cache_shape))
            self.register_buffer("v_cache", torch.zeros(cache_shape))

        def update(self, input_pos, k_val, v_val):
            # Simple scatter via indexing (no custom ops)
            seq_len = k_val.shape[1]
            for i in range(seq_len):
                pos = input_pos[i].item()
                self.k_cache[0, pos] = k_val[0, i]
                self.v_cache[0, pos] = v_val[0, i]
            return self.k_cache, self.v_cache

    class EagerSDPA(torch.nn.Module):
        def __init__(self, n_heads, n_kv_heads, head_dim):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = head_dim
            self.dim = n_heads * head_dim
            self.repeats = n_heads // n_kv_heads

        def forward(self, input_pos, q, k_cache, v_cache, bsz, seqlen, mask=None):
            start_pos = input_pos[0].item()
            kv_len = start_pos + seqlen

            q = q.transpose(1, 2)
            k = k_cache[:, :kv_len, :, :].transpose(1, 2)
            v = v_cache[:, :kv_len, :, :].transpose(1, 2)

            if self.repeats > 1:
                k = k.repeat_interleave(self.repeats, dim=1)
                v = v.repeat_interleave(self.repeats, dim=1)

            q = q.float()
            k = k.float()
            v = v.float()
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(seqlen > 1))
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
            return y

    for name, module in model.named_modules():
        if hasattr(module, 'sdpa') and isinstance(module.sdpa, SDPA):
            n_kv = module.n_kv_heads
            module.sdpa = EagerSDPA(module.n_heads, n_kv, module.head_dim)
        if hasattr(module, 'kv_cache') and isinstance(module.kv_cache, KVCache):
            old_cache = module.kv_cache
            new_cache = EagerKVCache(
                old_cache.k_cache.shape[1],
                old_cache.k_cache.shape[2],
                old_cache.k_cache.shape[3],
            )
            module.kv_cache = new_cache


def write_wav(path: str, samples: torch.Tensor, sample_rate: int = 24000):
    samples = samples.squeeze().float().cpu()
    samples = samples.clamp(-1.0, 1.0)
    n = samples.numel()
    data_size = n * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        pcm = (samples * 32767).to(torch.int16)
        f.write(pcm.numpy().tobytes())


def tokenize_text(tokenizer_path: str, text: str) -> list[int]:
    """Tokenize text using the Tekken tokenizer (mistral_common)."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    tok = MistralTokenizer.from_file(tokenizer_path)
    inner = tok.instruct_tokenizer.tokenizer
    return inner.encode(text, bos=False, eos=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", default="Hello, how are you today?")
    parser.add_argument("--voice", default=None,
                        help="Voice name or path to .pt file")
    parser.add_argument("--output", default="/tmp/voxtral_eager.wav")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Semantic sampling temperature (0=greedy)")
    parser.add_argument(
        "--trace-json",
        default=None,
        help="Optional path to write a structured parity trace JSON.",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    model_dir = Path(args.model_path)

    # Load model in FP32 and swap out export-only custom ops for eager-safe
    # implementations before using the result as a parity oracle.
    print("Loading model in FP32 eager mode...")
    model = load_model(args.model_path, max_seq_len=4096, dtype=torch.float32)
    _patch_eager_sdpa(model)
    config = model.config

    # Zero all KV caches after patching so the eager fallback starts from a
    # clean cache state as well.
    for layer in model.decoder.layers:
        layer.attention.kv_cache.k_cache.zero_()
        layer.attention.kv_cache.v_cache.zero_()
    print("  Patched eager SDPA/KV cache and zeroed caches")

    # Load voice embedding using the same resolution rules we want elsewhere:
    # default neutral_female, prefer .pt, and allow raw BF16 .bin.
    voice_embed, voice_path = load_voice_from_model_dir(model_dir, args.voice, dim=config.dim)
    voice_name = voice_path.stem
    print(f"Loading voice from {voice_path}")
    voice_len = voice_embed.shape[0]
    print(f"  Voice: {voice_embed.shape} ({voice_embed.dtype})")

    # Tokenize text
    tokenizer_path = str(model_dir / "tekken.json")
    text_tokens = tokenize_text(tokenizer_path, args.text)
    print(f"  Text tokens: {len(text_tokens)}")

    prompt = build_reference_prompt_ids(
        text_tokens=text_tokens,
        voice_len=voice_len,
        begin_audio_token_id=config.begin_audio_token_id,
        audio_token_id=config.audio_token_id,
        text_to_audio_token_id=config.text_to_audio_token_id,
        repeat_audio_text_token_id=config.repeat_audio_text_token_id,
    )
    official_prompt_ids = encode_speech_request_tokens(tokenizer_path, args.text, voice_name)
    if prompt.token_ids != official_prompt_ids:
        raise RuntimeError(
            "Manual prompt construction diverges from mistral_common "
            f"encode_speech_request for voice={voice_name}"
        )

    prompt_len = len(official_prompt_ids)
    print(f"  Prompt: {prompt_len} tokens (voice_start={prompt.voice_start}, "
          f"voice_len={prompt.voice_len}, text={len(text_tokens)})")

    trace: dict[str, object] = {
        "mode": "eager_reference",
        "text": args.text,
        "voice_path": str(voice_path),
        "prompt_token_ids": official_prompt_ids,
        "voice_start": prompt.voice_start,
        "voice_len": prompt.voice_len,
        "frames": [],
    }

    with torch.no_grad():
        # Embed prompt tokens
        prompt_t = torch.tensor([official_prompt_ids], dtype=torch.long)
        embeds = model.decoder.tok_embeddings(prompt_t)  # (1, prompt_len, 3072)

        # Splice voice embeddings over AUDIO placeholders
        embeds = splice_voice_embeddings(embeds, voice_embed, prompt.voice_start)
        print("  Voice spliced into prompt embeddings")

        print("Prefilling decoder + running AUDIO seed...")
        t0 = time.time()
        seed_trace = run_seed_decode(
            token_embedding=model.decoder.tok_embeddings,
            decoder=model.decoder,
            audio_token_id=config.audio_token_id,
            prompt_embeds=embeds,
        )
        print(f"  Prefill + seed done in {time.time()-t0:.1f}s")

        hidden = seed_trace.seed_hidden  # (1, 3072)
        cur_pos = seed_trace.seed_position + 1
        print(f"  Prefill hidden norm: {seed_trace.prefill_hidden.norm().item():.4f}")
        print(f"  Seed hidden norm: {hidden.norm().item():.4f}")

        trace["prefill_hidden"] = seed_trace.prefill_hidden[0].float().tolist()
        trace["frame0_hidden"] = hidden[0].float().tolist()
        trace["seed_hidden"] = hidden[0].float().tolist()
        trace["seed_position"] = seed_trace.seed_position
        trace["seed_step_applied"] = True
        trace["frame0_position"] = seed_trace.seed_position

        # Autoregressive generation
        print(f"Generating audio (max {args.max_frames} frames)...")
        gen = torch.Generator()
        gen.manual_seed(args.seed)

        all_codes = []
        n_steps = config.n_decoding_steps
        timesteps = torch.linspace(0, 1, n_steps + 1)
        t_gen_start = time.time()

        for frame in range(args.max_frames):
            # Semantic head
            raw_logits = model.flow_head.semantic_codebook_output(hidden).float()
            raw_logits[:, EMPTY_AUDIO_ID] = float("-inf")
            raw_logits[:, (N_SPECIAL_TOKENS + config.semantic_codebook_size):] = float("-inf")

            if args.temperature > 0:
                probs = torch.softmax(raw_logits / args.temperature, dim=-1)
                semantic_code = torch.multinomial(probs, 1).squeeze(-1)
            else:
                semantic_code = raw_logits.argmax(dim=-1)
            code_val = semantic_code.item()

            top5 = topk_pairs(raw_logits[0], k=5)
            if frame < 5:
                formatted_top5 = [
                    (item["id"], f"{item['logit']:.2f}") for item in top5
                ]
                print(f"  [logits] top5: {formatted_top5}")

            if code_val == END_AUDIO_ID:
                if frame < 3:
                    trace["frames"].append(
                        {
                            "frame": frame,
                            "hidden_norm_before_frame": float(hidden.norm().item()),
                            "semantic_code": int(code_val),
                            "semantic_topk": top5,
                            "full_codes": [],
                            "end_audio": True,
                        }
                    )
                trace["end_audio_at_frame"] = frame
                print(f"\n  END_AUDIO at frame {frame}")
                break

            # Flow matching ODE (7 Euler steps with CFG)
            x = torch.randn(1, config.acoustic_dim, generator=gen)
            x = x * config.noise_scale
            llm_zero = torch.zeros_like(hidden)

            for step in range(n_steps):
                t = timesteps[step]
                dt = timesteps[step + 1] - timesteps[step]
                t_idx = torch.tensor([step], dtype=torch.long)

                v_cond = model.flow_head.predict_velocity(x, t_idx, hidden)
                v_uncond = model.flow_head.predict_velocity(x, t_idx, llm_zero)
                v = config.cfg_alpha * v_cond + (1 - config.cfg_alpha) * v_uncond
                x = x + v * dt

            # Quantize acoustic codes
            x_clamped = torch.clamp(x, -1, 1)
            scaled = ((x_clamped + 1) / 2) * (config.acoustic_levels - 1)
            acoustic_codes = scaled.round().long() + N_SPECIAL_TOKENS

            # Full frame: [semantic, acoustic_0, ..., acoustic_35]
            frame_codes = torch.cat([
                semantic_code.view(1, 1),
                acoustic_codes,
            ], dim=1)  # (1, 37)
            all_codes.append(frame_codes)
            if frame == 0:
                trace["frame0_full_codes"] = frame_codes[0].tolist()

            if frame < 3:
                x_final = x_clamped[0]
                print(f"  [flow] x range=[{x_final.min():.4f}, {x_final.max():.4f}], "
                      f"codes: {acoustic_codes[0, :6].tolist()}")

            if frame < 3:
                trace["frames"].append(
                    {
                        "frame": frame,
                        "hidden_norm_before_frame": float(hidden.norm().item()),
                        "semantic_code": int(code_val),
                        "semantic_topk": top5,
                        "full_codes": frame_codes[0].tolist(),
                        "x_min": float(x_clamped.min().item()),
                        "x_max": float(x_clamped.max().item()),
                    }
                )

            # Feed back through audio token embedding
            codes_for_embed = frame_codes.unsqueeze(-1)  # (1, 37, 1)
            next_embed = model.audio_token_embedding(codes_for_embed)  # (1, 1, 3072)
            if frame == 0:
                trace["frame0_audio_embed"] = next_embed[0, 0].float().tolist()

            next_pos = torch.tensor([cur_pos], dtype=torch.long)
            hidden = model.decoder(next_embed, next_pos)  # (1, 1, 3072)
            hidden = hidden[:, 0, :]  # (1, 3072)
            if frame == 0:
                trace["frame1_position"] = int(next_pos.item())
                trace["frame1_hidden"] = hidden[0].float().tolist()
            cur_pos += 1

            elapsed = time.time() - t_gen_start
            audio_sec = (frame + 1) / 12.5
            if frame < 5 or (frame + 1) % 10 == 0:
                print(f"  Frame {frame+1}: sem={code_val}, "
                      f"h_norm={hidden.norm().item():.1f}, "
                      f"audio={audio_sec:.1f}s, elapsed={elapsed:.1f}s")

        gen_elapsed = time.time() - t_gen_start
        n_frames = len(all_codes)
        if n_frames == 0:
            trace["generated_frames"] = 0
            trace["waveform"] = {
                "shape": [1, 1, 0],
                "min": 0.0,
                "max": 0.0,
                "mean_abs": 0.0,
                "peak_abs": 0.0,
            }
            if args.trace_json:
                Path(args.trace_json).write_text(
                    json.dumps(trace, indent=2, sort_keys=True) + "\n"
                )
                print(f"  Wrote trace JSON: {args.trace_json}")
            print("ERROR: No audio frames generated")
            sys.exit(1)

        audio_duration = n_frames / 12.5
        print(f"\n  Generated {n_frames} frames ({audio_duration:.1f}s audio) "
              f"in {gen_elapsed:.1f}s (RTF={gen_elapsed/audio_duration:.2f})")

        # Codec decode
        print("Running codec decoder...")
        codes_tensor = torch.stack(all_codes, dim=2)  # (1, 37, n_frames)
        print(f"  Codes shape: {codes_tensor.shape}")

        t_codec = time.time()
        waveform = model.codec_decoder(codes_tensor)  # (1, 1, n_frames*1920)
        print(f"  Codec done in {time.time()-t_codec:.1f}s")
        print(f"  Waveform: {waveform.shape}, range: [{waveform.min():.4f}, {waveform.max():.4f}]")

        trace["generated_frames"] = n_frames
        trace["waveform"] = {
            "shape": list(waveform.shape),
            "min": float(waveform.min().item()),
            "max": float(waveform.max().item()),
            "mean_abs": float(waveform.abs().mean().item()),
            "peak_abs": float(waveform.abs().max().item()),
        }

    # Write WAV
    write_wav(args.output, waveform, config.sampling_rate)
    print(f"\nWrote {args.output} "
          f"({waveform.numel() / config.sampling_rate:.1f}s, "
          f"{config.sampling_rate}Hz)")

    # Quick amplitude check
    amp = waveform.abs().mean().item()
    peak = waveform.abs().max().item()
    print(f"  Mean amplitude: {amp:.6f}, Peak: {peak:.6f}")
    if peak < 0.001:
        print("  WARNING: Very low amplitude - likely silence")

    if args.trace_json:
        Path(args.trace_json).write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n"
        )
        print(f"  Wrote trace JSON: {args.trace_json}")


if __name__ == "__main__":
    main()
