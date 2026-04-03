"""Profile Qwen3.5 MoE inference with torch.profiler for operator-level breakdown."""

import argparse
import os
import time

import torch

from executorch.examples.models.qwen3_5_moe.inference import _move_to_cuda, _sample
from executorch.examples.models.qwen3_5_moe.export import load_prequantized_model


def generate_with_profile(
    model, tokenizer, prompt, max_new_tokens=16, temperature=0.0, eos_token_ids=None,
    warmup_tokens=0,
):
    """Generate text and profile the decode phase."""
    if eos_token_ids is None:
        eos_token_ids = set()

    input_ids = tokenizer.encode(prompt).ids

    # === Prefill (no profiling, just get through it) ===
    print(f"Prefilling {len(input_ids)} tokens...")
    with torch.no_grad():
        for i, tok_id in enumerate(input_ids):
            tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            logits = model(tok, pos)

    # Sample first token
    next_token = _sample(logits[:, -1, :], temperature)
    generated = [next_token.item()]
    seq_len = len(input_ids)

    # === Warmup decode tokens (no profiling) ===
    print(f"Warming up {warmup_tokens} decode tokens...")
    with torch.no_grad():
        for i in range(warmup_tokens):
            pos = torch.tensor([seq_len + i], device="cuda")
            logits = model(next_token.unsqueeze(0), pos)
            next_token = _sample(logits[:, -1, :], temperature)
            tok_id = next_token.item()
            generated.append(tok_id)
            if tok_id in eos_token_ids:
                print("  (hit EOS during warmup)")
                return tokenizer.decode(generated), None

    torch.cuda.synchronize()

    # === Profiled decode tokens ===
    print(f"Profiling {max_new_tokens} decode tokens...")
    profile_start_idx = warmup_tokens

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for i in range(max_new_tokens):
                idx = profile_start_idx + i
                pos = torch.tensor([seq_len + idx], device="cuda")
                logits = model(next_token.unsqueeze(0), pos)
                next_token = _sample(logits[:, -1, :], temperature)
                tok_id = next_token.item()
                generated.append(tok_id)
                if tok_id in eos_token_ids:
                    print("  (hit EOS during profiling)")
                    break

    torch.cuda.synchronize()
    return tokenizer.decode(generated), prof


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen3.5 MoE inference")
    parser.add_argument(
        "--prequantized",
        required=True,
        help="Path to prequantized bundle directory",
    )
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Number of decode tokens to profile")
    parser.add_argument("--warmup-tokens", type=int, default=4,
                        help="Number of decode tokens to run before profiling")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--trace-output", type=str, default=None,
                        help="Path to save Chrome trace JSON (optional)")
    parser.add_argument("--row-limit", type=int, default=50,
                        help="Number of rows in profiler table")
    args = parser.parse_args()

    # Register Triton kernels
    import executorch.backends.cuda.triton.kernels  # noqa: F401

    # Load model
    print(f"Loading model from {args.prequantized}...")
    model, config = load_prequantized_model(
        args.prequantized, max_seq_len=args.max_seq_len
    )
    _move_to_cuda(model, config)
    model.eval()

    # Compile
    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    # Load tokenizer
    tokenizer_path = os.path.join(args.prequantized, "tokenizer.json")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # EOS tokens
    eos_token_ids = set()
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        ids = tokenizer.encode(tok).ids
        if len(ids) == 1:
            eos_token_ids.add(ids[0])

    # Run profiled generation
    t0 = time.perf_counter()
    output, prof = generate_with_profile(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_tokens=args.warmup_tokens,
        temperature=args.temperature,
        eos_token_ids=eos_token_ids,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*80}")
    print(f"Generated text: {output}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"{'='*80}\n")

    if prof is not None:
        # Print CUDA time breakdown
        print("=" * 80)
        print("OPERATOR-LEVEL BREAKDOWN (sorted by CUDA time total)")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=args.row_limit
        ))

        print("\n" + "=" * 80)
        print("OPERATOR-LEVEL BREAKDOWN (sorted by CUDA time, grouped by input shape)")
        print("=" * 80)
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=args.row_limit
        ))

        # Save Chrome trace if requested
        if args.trace_output:
            prof.export_chrome_trace(args.trace_output)
            print(f"\nChrome trace saved to: {args.trace_output}")


if __name__ == "__main__":
    main()
