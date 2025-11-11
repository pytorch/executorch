# ============================================================================
# IMPORTANT: Import sdpa_triton BEFORE defining the model
# This automatically enables the custom Triton kernel via monkey-patching
# ============================================================================
import argparse
import os
from contextlib import nullcontext

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from optimized_sdpa_triton import optimized_triton_scaled_dot_product_attention
from sdpa_triton import triton_scaled_dot_product_attention
from torch.export import Dim, export
from torch.nn.attention import SDPBackend


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        # This is the ORIGINAL code - we're NOT changing it!
        # But it will automatically use our custom Triton kernel
        # because we imported sdpa_triton above
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        return out


sdpa_ctx = nullcontext()


# hacky method to replace system sdpa with my triton
def init_sdpa_kernel(custom_triton):
    global sdpa_ctx
    if custom_triton == "decomposed_kernel":
        sdpa_ctx = torch.nn.attention.sdpa_kernel([SDPBackend.MATH])
    elif custom_triton == "unoptimized_triton":
        torch.nn.functional.scaled_dot_product_attention = (
            triton_scaled_dot_product_attention
        )
    elif custom_triton == "optimized_triton":
        torch.nn.functional.scaled_dot_product_attention = (
            optimized_triton_scaled_dot_product_attention
        )
    else:
        assert False, f"{custom_triton} has not been supported yet"


def main(kernel_type, output_dir, dtype):
    print(f"Using kernel type: {kernel_type}")
    print(f"Using dtype: {dtype}")
    init_sdpa_kernel(kernel_type)

    model = Model()
    batch_size, num_heads, seq_len, head_dim = 1, 20, 1500, 64

    # Map dtype string to torch dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    # Create inputs with specified dtype
    inputs = (
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=torch_dtype,
            device="cuda",
        ),
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=torch_dtype,
            device="cuda",
        ),
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=torch_dtype,
            device="cuda",
        ),
    )

    print("Testing model execution with custom kernel...")
    with torch.no_grad():
        output = model(*inputs)
        print(f"✓ Model executed successfully. Output shape: {output.shape}\n")

    print("Exporting model...")
    exported_program = export(model, inputs)
    print("✓ Model exported successfully\n")

    print("Lowering to ExecuTorch CUDA backend (using AOTI)...")
    with sdpa_ctx, torch.no_grad():
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()
        print("✓ Model lowered successfully with AOTI\n")

        print("Saving model...")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model.pte"), "wb") as file:
            file.write(executorch_program.buffer)

        executorch_program.write_tensor_data_to_file(output_dir)
        print(f"✓ PTE and PTD files has successfully dumped to {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SDPA with custom kernel")
    parser.add_argument(
        "--kernel_type",
        type=str,
        choices=["unoptimized_triton", "optimized_triton", "decomposed_kernel"],
        help="Type of kernel to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save model.pte and tensor data (default: current directory)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help="Data type for model inputs (default: bf16)",
    )

    args = parser.parse_args()

    main(args.kernel_type, args.output_dir, args.dtype)
