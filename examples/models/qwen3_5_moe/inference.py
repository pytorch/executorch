"""Run inference on a prequantized Qwen 3.5 MoE model using torch.compile.

Loads the quantized model from a safetensors bundle, compiles with
torch.compile for fast CUDA inference, and generates text.

Usage:
  python inference.py --prequantized /path/to/bundle --prompt "Hello"
  python inference.py --prequantized /path/to/bundle --prompt "What is 2+2?" --max-new-tokens 64
"""

import argparse
import os
import time

import torch

from executorch.examples.models.qwen3_5_moe.export import load_prequantized_model


def _move_to_cuda(model, config):
    """Move model to CUDA, materializing meta buffers directly on device.

    Handles Int4TilePackedTo4dTensor and other tensor subclasses by moving
    parameters individually with explicit device transfer. Recomputes RoPE
    inv_freq and causal masks on CUDA.
    """
    for fqn, buf in list(model.named_buffers()):
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        if buf.device.type == "meta":
            dtype = torch.bfloat16 if buf.dtype != torch.bool else torch.bool
            parent.register_buffer(
                parts[-1], torch.zeros(buf.shape, dtype=dtype, device="cuda")
            )
        else:
            parent.register_buffer(parts[-1], buf.to("cuda"))

    for name, p in model.named_parameters():
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(
            parent,
            parts[-1],
            torch.nn.Parameter(p.data.to("cuda"), requires_grad=False),
        )

    # Recompute RoPE inv_freq on CUDA
    for layer in model.layers:
        if hasattr(layer.attn, "rotary_emb"):
            rope = layer.attn.rotary_emb
            inv_freq = 1.0 / (
                config.rope_theta
                ** (
                    torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
                    / rope.rotary_dim
                )
            )
            rope.inv_freq = inv_freq.to("cuda")
        if hasattr(layer.attn, "mask"):
            layer.attn.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(
                        config.max_seq_len,
                        config.max_seq_len,
                        dtype=torch.bool,
                        device="cuda",
                    )
                ),
            )


def generate(
    model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, eos_token_ids=None
):
    """Generate text autoregressively with KV cache.

    Prefills one token at a time (the chunk_gated_delta_rule kernel's chunked
    path has numerical issues with T>1 in eager mode; token-by-token uses the
    stable recurrent path).
    """
    if eos_token_ids is None:
        eos_token_ids = set()

    input_ids = tokenizer.encode(prompt).ids

    # Prefill: one token at a time
    with torch.no_grad():
        for i, tok_id in enumerate(input_ids):
            tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            logits = model(tok, pos)

    # Sample first generated token
    next_token = _sample(logits[:, -1, :], temperature)
    generated = [next_token.item()]

    # Decode: one token at a time
    seq_len = len(input_ids)
    with torch.no_grad():
        for i in range(max_new_tokens - 1):
            pos = torch.tensor([seq_len + i], device="cuda")
            logits = model(next_token.unsqueeze(0), pos)
            next_token = _sample(logits[:, -1, :], temperature)
            tok_id = next_token.item()
            generated.append(tok_id)
            if tok_id in eos_token_ids:
                break

    return tokenizer.decode(generated)


def _sample(logits, temperature):
    """Sample from logits with temperature."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on prequantized Qwen3.5 MoE"
    )
    parser.add_argument(
        "--prequantized",
        required=True,
        help="Path to prequantized bundle directory",
    )
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0=greedy)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=32768, help="KV cache length"
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    args = parser.parse_args()

    # Register Triton kernels (fused MoE, GatedDeltaNet)
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

    # EOS tokens for Qwen: <|im_end|> and <|endoftext|>
    eos_token_ids = set()
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        ids = tokenizer.encode(tok).ids
        if len(ids) == 1:
            eos_token_ids.add(ids[0])

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    t0 = time.perf_counter()
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_ids=eos_token_ids,
    )
    elapsed = time.perf_counter() - t0

    print(output)
    print("-" * 40)
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
