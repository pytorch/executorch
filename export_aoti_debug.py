"""
Export Qwen3.5 MoE via direct AOTI with configurable num_layers.
For debugging torch.cond correctness with fewer layers.
"""

import os
import sys
import time

import torch
import torch._inductor.config as inductor_config

# Register FLA Triton kernel
import executorch.backends.cuda.triton.kernels  # noqa: F401
from executorch.examples.models.qwen3_5_moe.export import (
    load_prequantized_model,
    _materialize_buffers,
)


def main():
    prequantized_dir = "/home/gasoonjia/models/Qwen3.5-35B-A3B-HQQ-INT4-local/"
    num_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    # Inductor config
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    inductor_config.allow_buffer_reuse = False

    # Load model
    model, config = load_prequantized_model(prequantized_dir)
    _materialize_buffers(model, config)

    # Trim to num_layers
    if num_layers < config.num_hidden_layers:
        model.layers = model.layers[:num_layers]
        config.num_hidden_layers = num_layers
        print(f"Trimmed to {num_layers} layers")
        # Count how many are GatedDeltaNet (have torch.cond)
        n_gdn = sum(1 for l in model.layers if l.layer_type == "linear_attention")
        n_fa = sum(1 for l in model.layers if l.layer_type == "full_attention")
        print(f"  {n_gdn} GatedDeltaNet (torch.cond), {n_fa} FullAttention")

    model.to("cuda")
    model.eval()

    # Export
    from torch.export import Dim, export

    example_tokens = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device="cuda")
    example_input_pos = torch.arange(5, device="cuda")
    seq_dim = Dim("seq_len", min=1, max=config.max_seq_len - 1)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting...")
    t0 = time.time()
    with torch.no_grad():
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print(f"Export done in {time.time()-t0:.1f}s")

    print("AOTI compiling...")
    t0 = time.time()
    so_path = torch._inductor.aot_compile(
        exported.module(),
        (example_tokens, example_input_pos),
    )
    print(f"AOTI compile done in {time.time()-t0:.1f}s")
    print(f"Compiled .so: {so_path}")

    # Validate single forward pass from zero state
    runner = torch._export.aot_load(so_path, "cuda")

    for T in [1, 4, 7]:
        tokens = torch.randint(0, config.vocab_size, (1, T), device="cuda")
        input_pos = torch.arange(T, device="cuda")
        path = "recurrent" if T < 4 else "chunked"

        # Reset model state
        for name, buf in model.named_buffers():
            if any(k in name for k in ('k_cache', 'v_cache', 'conv_state', 'recurrent_state')):
                buf.zero_()

        # Eager
        with torch.no_grad():
            eager_out = model(tokens, input_pos)

        # Fresh AOTI runner (reload to reset internal state)
        runner = torch._export.aot_load(so_path, "cuda")
        with torch.no_grad():
            aoti_out = runner(tokens, input_pos)

        diff = (eager_out.float() - aoti_out.float()).abs().max().item()
        has_nan = aoti_out.isnan().any().item()
        eager_top5 = eager_out[:, -1].topk(5).indices[0].tolist()
        aoti_top5 = aoti_out[:, -1].topk(5).indices[0].tolist()
        overlap = len(set(eager_top5) & set(aoti_top5))
        ok = "PASS" if (diff < 2.0 and overlap >= 3 and not has_nan) else "FAIL"
        print(f"T={T:>3} ({path:>9}): max_diff={diff:.4f} top5_overlap={overlap}/5 {ok}")
        if overlap < 3:
            print(f"  eager: {eager_top5}")
            print(f"  aoti:  {aoti_top5}")


if __name__ == "__main__":
    main()
