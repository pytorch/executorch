#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gemma 3 Multimodal Python Binding Example

This script demonstrates how to run Gemma 3 multimodal inference using
ExecuTorch Python bindings. It loads a .pte model and processes both
text and image inputs.

Example usage:
    python pybinding_run.py \
        --model_path /path/to/model.pte \
        --tokenizer_path /path/to/tokenizer.json \
        --image_path /path/to/image.png \
        --prompt "What is in this image?"

Requirements:
    - ExecuTorch with Python bindings installed
    - PIL (Pillow) for image loading
    - numpy for array operations
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image

# Load required operator libraries for quantized and custom ops
# These must be imported BEFORE creating the runner to register the operators
try:
    import executorch.kernels.quantized  # noqa: F401
except Exception as e:
    print(f"Warning: Failed to load quantized kernels: {e}")
    print("The model may fail if it uses quantized operators.")
    print("To fix this, reinstall ExecuTorch with: pip install executorch")

try:
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
except Exception as e:
    print(f"Warning: Failed to load custom ops: {e}")
    print("The model may fail if it uses custom operators like custom_sdpa.")

from executorch.extension.llm.runner import (
    GenerationConfig,
    make_image_input,
    make_text_input,
    MultimodalRunner,
)


def load_image(image_path: str, target_size: int = 896) -> torch.Tensor:
    """
    Load and preprocess an image for Gemma 3 vision encoder.

    The image is:
    1. Loaded and converted to RGB
    2. Resized to target_size x target_size (default 896x896)
    3. Converted from HWC to CHW format
    4. Normalized from uint8 [0, 255] to float32 [0.0, 1.0]

    Args:
        image_path: Path to the image file (.jpg, .png, .bmp, etc.)
        target_size: Target size for resizing (default 896 for Gemma 3)

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (3, target_size, target_size)
    """
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((target_size, target_size))

    # Convert to tensor: HWC -> CHW, uint8 -> float32 [0, 1]
    image_tensor = (
        torch.from_numpy(np.array(pil_image))
        .permute(2, 0, 1)
        .contiguous()
        .float()
        / 255.0
    )

    return image_tensor


def build_multimodal_inputs(prompt: str, image_tensor: torch.Tensor) -> list:
    """
    Build the multimodal input sequence for Gemma 3.

    The input format follows the Gemma 3 chat template:
    <start_of_turn>user
    <start_of_image>[IMAGE]{prompt}<end_of_turn>
    <start_of_turn>model

    Args:
        prompt: The text prompt/question about the image
        image_tensor: Preprocessed image tensor from load_image()

    Returns:
        list: List of MultimodalInput objects for the runner
    """
    inputs = []
    inputs.append(make_text_input("<start_of_turn>user\n<start_of_image>"))
    inputs.append(make_image_input(image_tensor))
    inputs.append(make_text_input(f"{prompt}<end_of_turn>\n<start_of_turn>model\n"))
    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma 3 multimodal inference with ExecuTorch Python bindings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python pybinding_run.py --model_path model.pte --tokenizer_path tokenizer.json \\
        --image_path image.png --prompt "What is in this image?"

    # With custom generation settings
    python pybinding_run.py --model_path model.pte --tokenizer_path tokenizer.json \\
        --image_path image.png --prompt "Describe this image in detail" \\
        --max_new_tokens 200 --temperature 0.7
        """,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .pte model file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer.json file",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is in this image?",
        help="Text prompt for the model (default: 'What is in this image?')",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 for greedy decoding (default: 0.0)",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Loading tokenizer from: {args.tokenizer_path}")

    # Create the multimodal runner
    runner = MultimodalRunner(args.model_path, args.tokenizer_path)

    # Load and preprocess the image
    print(f"Loading image from: {args.image_path}")
    image_tensor = load_image(args.image_path)
    print(f"Image tensor shape: {image_tensor.shape}")

    # Build multimodal inputs with Gemma 3 chat template
    inputs = build_multimodal_inputs(args.prompt, image_tensor)

    # Configure generation settings
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        echo=False,
    )

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    print("Response: ", end="", flush=True)

    # Collect generated tokens
    # Note: The C++ MultimodalRunner already prints tokens via safe_printf(),
    # so we don't print in the callback to avoid duplication
    generated_tokens = []
    stop_generation = False

    def token_callback(token: str):
        nonlocal stop_generation
        # Stop collecting after first <end_of_turn> token
        if stop_generation:
            return
        if "<end_of_turn>" in token:
            # Add any text before the end token
            before_end = token.split("<end_of_turn>")[0]
            if before_end:
                generated_tokens.append(before_end)
            stop_generation = True
            return
        generated_tokens.append(token)

    def stats_callback(stats):
        # Print the complete response (since C++ prints token by token)
        print()  # Newline after streaming output
        print("-" * 50)
        print(f"Prompt tokens: {stats.num_prompt_tokens}")
        print(f"Generated tokens: {stats.num_generated_tokens}")
        # Calculate time to first token
        time_to_first_token_s = (stats.first_token_ms - stats.inference_start_ms) / 1000.0
        print(f"Time to first token: {time_to_first_token_s:.3f} s")
        # Calculate generation rate
        generation_time_s = (stats.inference_end_ms - stats.first_token_ms) / 1000.0
        if generation_time_s > 0:
            tokens_per_sec = stats.num_generated_tokens / generation_time_s
            print(f"Generation rate: {tokens_per_sec:.2f} tokens/sec")

    # Run generation
    runner.generate(inputs, config, token_callback, stats_callback)

    return 0


if __name__ == "__main__":
    sys.exit(main())
