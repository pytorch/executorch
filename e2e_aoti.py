"""End-to-end AOTI inference: prefill (chunked) + decode (recurrent) via torch.cond."""

import os
import sys
import time

import torch
import torch._inductor.config as inductor_config

# Register Triton kernels
import executorch.backends.cuda.triton.kernels  # noqa: F401
from executorch.examples.models.qwen3_5_moe.export import (
    load_prequantized_model,
    _materialize_buffers,
)


def main():
    so_path = sys.argv[1] if len(sys.argv) > 1 else None
    prequantized_dir = "/home/gasoonjia/models/Qwen3.5-35B-A3B-HQQ-INT4-local/"
    tokenizer_path = "/home/gasoonjia/models/Qwen3.5-35B-A3B/tokenizer.json"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is 2+2?"
    max_new_tokens = 64

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # EOS tokens
    eos_token_ids = set()
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        ids = tokenizer.encode(tok).ids
        if len(ids) == 1:
            eos_token_ids.add(ids[0])

    input_ids = tokenizer.encode(prompt).ids
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(input_ids)}")

    if so_path is None:
        # Compile from scratch
        print("No .so provided, compiling...")
        inductor_config.coordinate_descent_tuning = False
        inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
        inductor_config.allow_buffer_reuse = False

        model, config = load_prequantized_model(prequantized_dir)
        _materialize_buffers(model, config)
        model.to("cuda").eval()

        from torch.export import Dim, export
        example_tokens = torch.tensor([[0, 1]], dtype=torch.long, device="cuda")
        example_input_pos = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        seq_dim = Dim("seq_len", min=1, max=config.max_seq_len - 1)
        dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

        t0 = time.time()
        with torch.no_grad():
            exported = export(model, (example_tokens, example_input_pos),
                            dynamic_shapes=dynamic_shapes, strict=True)
        print(f"Export: {time.time()-t0:.1f}s")

        t0 = time.time()
        so_path = torch._inductor.aot_compile(exported.module(),
                                               (example_tokens, example_input_pos))
        print(f"Compile: {time.time()-t0:.1f}s")
        print(f".so: {so_path}")
        del model, exported
        torch.cuda.empty_cache()
    else:
        print(f"Loading .so: {so_path}")

    # Load AOTI runner
    runner = torch._export.aot_load(so_path, "cuda")

    # --- Prefill (chunked path, T >= 4) ---
    prefill_len = len(input_ids)
    tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    input_pos = torch.arange(prefill_len, dtype=torch.long, device="cuda")

    print(f"\nPrefill ({prefill_len} tokens)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = runner(tokens, input_pos)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    print(f"Prefill: {prefill_time*1000:.1f}ms")

    # Sample first token
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = [next_token.item()]

    # --- Decode (recurrent path, T=1) ---
    print(f"Decoding up to {max_new_tokens} tokens...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(max_new_tokens - 1):
            pos = torch.tensor([prefill_len + i], dtype=torch.long, device="cuda")
            logits = runner(next_token.unsqueeze(0), pos)
            next_token = logits[:, -1, :].argmax(dim=-1)
            tok_id = next_token.item()
            generated.append(tok_id)
            if tok_id in eos_token_ids:
                break
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    output = tokenizer.decode(generated)
    n_decode = len(generated)

    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"{'='*50}")
    print(f"Prefill: {prefill_time*1000:.1f}ms ({prefill_len} tokens)")
    print(f"Decode:  {decode_time*1000:.1f}ms ({n_decode} tokens, "
          f"{n_decode/decode_time:.1f} tok/s)")


if __name__ == "__main__":
    main()
