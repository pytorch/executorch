#!/usr/bin/env python3
"""
Create binary input files for executor_runner.

This script tokenizes text and saves the input_ids and attention_mask
as binary files that can be used with executor_runner.

Usage:
    python create_input_bins.py
    python create_input_bins.py --text "Your custom text"
    python create_input_bins.py --text "Hello world" --output-dir ./my_inputs
"""

import argparse
import os

from transformers import AutoTokenizer


def create_input_bins(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 128,
    output_dir: str = "./",
):
    """
    Tokenize text and save as binary files for executor_runner.

    Args:
        text: Input text to tokenize
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length
        output_dir: Directory to save binary files
    """
    print("=" * 80)
    print("Creating Input Binary Files for executor_runner")
    print("=" * 80)
    print(f"\nText: {text}")
    print(f"Model: {model_name}")
    print(f"Max Length: {max_length}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")

    # Tokenize text
    print("\n[2/3] Tokenizing text...")
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    print(f"✓ Tokenized")
    print(f"  - input_ids shape: {list(encoded['input_ids'].shape)}")
    print(f"  - attention_mask shape: {list(encoded['attention_mask'].shape)}")
    print(f"  - First 10 tokens: {encoded['input_ids'][0][:10].tolist()}")

    # Save binary files
    print(f"\n[3/3] Saving binary files to {output_dir}/...")

    input_ids_file = os.path.join(output_dir, "input_ids.bin")
    attention_mask_file = os.path.join(output_dir, "attention_mask.bin")

    encoded["input_ids"].numpy().tofile(input_ids_file)
    encoded["attention_mask"].numpy().tofile(attention_mask_file)

    print(f"✓ Saved input_ids to: {input_ids_file}")
    print(f"✓ Saved attention_mask to: {attention_mask_file}")

    # Print next steps
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("\nRun executor_runner with these inputs:")
    print(f"\n  ./cmake-out/executor_runner \\")
    print(f"      --model_path=./xnnpack_model/model.pte \\")
    print(f"      --inputs={input_ids_file},{attention_mask_file}")

    print("\nOptionally, save the output:")
    print(f"\n  ./cmake-out/executor_runner \\")
    print(f"      --model_path=./xnnpack_model/model.pte \\")
    print(f"      --inputs={input_ids_file},{attention_mask_file} \\")
    print(f"      --output_file=./output")

    print("\n" + "=" * 80)
    print("✓ Done!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create binary input files for executor_runner"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an example sentence.",
        help="Text to tokenize (default: 'This is an example sentence.')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for tokenizer (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output directory for binary files (default: ./)",
    )

    args = parser.parse_args()

    create_input_bins(
        text=args.text,
        model_name=args.model,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
