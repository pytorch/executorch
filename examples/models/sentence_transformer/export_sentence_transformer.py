#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export all-MiniLM-L6-v2 model to ExecuTorch.

This script exports the all-MiniLM-L6-v2 sentence-transformer model using 
torch.export and ExecuTorch APIs. Supports XNNPack and CPU backends for 
optimized inference.

Example usage:
    # Export with XNNPack backend (recommended)
    python export_sentence_transformer.py --backend xnnpack

    # Export for CPU
    python export_sentence_transformer.py --backend cpu
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from .model import SentenceTransformerModel
except ImportError:
    # If running as a script, import from the same directory
    from model import SentenceTransformerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_with_xnnpack(
    model: torch.nn.Module,
    example_inputs: tuple,
    output_path: str,
):
    """Export model with XNNPack backend for optimized CPU inference."""
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower
    except ImportError as e:
        logger.error(f"Failed to import ExecuTorch XNNPack backend: {e}")
        raise

    logger.info("Exporting model to torch.export format...")

    # Export to torch.export
    exported_program = torch.export.export(
        model,
        example_inputs,
    )

    logger.info("Lowering to XNNPack backend...")

    # Lower to Edge IR with XNNPack partitioner
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    # Convert to ExecuTorch program
    logger.info("Converting to ExecuTorch format...")
    executorch_program = edge_program.to_executorch()

    # Save the model
    with open(output_path, "wb") as f:
        executorch_program.write_to_file(f)

    logger.info(f"Model saved to {output_path}")


def export_cpu(
    model: torch.nn.Module,
    example_inputs: tuple,
    output_path: str,
):
    """Export model for CPU execution (no backend delegation)."""
    try:
        from executorch.exir import to_edge
    except ImportError as e:
        logger.error(f"Failed to import ExecuTorch: {e}")
        raise

    logger.info("Exporting model to torch.export format...")

    # Export to torch.export
    exported_program = torch.export.export(
        model,
        example_inputs,
    )

    logger.info("Converting to Edge IR...")

    # Convert to Edge IR (no partitioning)
    edge_program = to_edge(exported_program)

    # Convert to ExecuTorch program
    logger.info("Converting to ExecuTorch format...")
    executorch_program = edge_program.to_executorch()

    # Save the model
    with open(output_path, "wb") as f:
        executorch_program.write_to_file(f)

    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export all-MiniLM-L6-v2 to ExecuTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with XNNPack backend (recommended)
  python export_sentence_transformer.py --backend xnnpack

  # Export with custom output directory
  python export_sentence_transformer.py --backend xnnpack --output-dir ./my_model

  # Export with CPU backend
  python export_sentence_transformer.py --backend cpu

  # Export with custom max sequence length
  python export_sentence_transformer.py --backend xnnpack --max-seq-length 256

Note: This example currently supports sentence-transformers/all-MiniLM-L6-v2
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="xnnpack",
        choices=["xnnpack", "cpu"],
        help="Backend to use for export (default: xnnpack)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./all_minilm_l6_v2_export",
        help="Output directory for exported model (default: ./all_minilm_l6_v2_export)",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model.pte"

    logger.info(f"Loading model: {args.model}")

    # Load model using our custom wrapper
    model = SentenceTransformerModel(args.model)
    model.eval()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create example inputs with static shapes
    logger.info(f"Creating example inputs (max_length={args.max_seq_length})...")
    text = "This is an example sentence for generating embeddings."
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=args.max_seq_length,
        truncation=True,
        return_tensors="pt",
    )

    example_inputs = (encoded["input_ids"], encoded["attention_mask"])

    logger.info(f"Exporting with {args.backend} backend...")

    # Export based on backend choice
    if args.backend == "xnnpack":
        export_with_xnnpack(model, example_inputs, str(output_path))
    else:
        export_cpu(model, example_inputs, str(output_path))

    logger.info(f"\n{'='*60}")
    logger.info("Export successful!")
    logger.info(f"{'='*60}")
    logger.info(f"Model file: {output_path}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"\nTo use this model:")
    logger.info(f"  1. Load the model from {output_path}")
    logger.info(f"  2. Tokenize your text with max_length={args.max_seq_length}")
    logger.info(f"  3. Pass input_ids and attention_mask to the model")
    logger.info(f"  4. Output will be sentence embeddings")


if __name__ == "__main__":
    main()
