#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare embeddings from the exported ExecuTorch model with the original transformers model.

This script validates that the exported model produces correct embeddings by:
1. Running inference with the original transformers model
2. Running inference with the exported ExecuTorch model (XNNPack or CPU backend)
3. Computing similarity metrics (cosine similarity, L2 distance)
4. Reporting the results

Example usage:
    # Compare embeddings from XNNPack model
    python compare_embeddings.py \
        --model-path sentence_transformer_export/model.pte \
        --model-name sentence-transformers/all-MiniLM-L6-v2

    # Compare embeddings from CPU model
    python compare_embeddings.py \
        --model-path exported_model_cpu/model.pte \
        --model-name sentence-transformers/all-MiniLM-L6-v2

    # Compare multiple sentences
    python compare_embeddings.py \
        --model-path sentence_transformer_export/model.pte \
        --model-name sentence-transformers/all-MiniLM-L6-v2 \
        --sentences "Hello world" "This is a test" "Machine learning is great"

    # Test different model (L12)
    python compare_embeddings.py \
        --model-path exported_l12/model.pte \
        --model-name sentence-transformers/all-MiniLM-L12-v2
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class OriginalSentenceTransformer:
    """Original sentence transformer model using transformers library."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: str, max_length: int = 128) -> np.ndarray:
        """Encode a sentence to embedding."""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = self.model(**encoded)
            token_embeddings = model_output.last_hidden_state
            sentence_embedding = self.mean_pooling(
                token_embeddings, encoded["attention_mask"]
            )

        return sentence_embedding.numpy()


class ExecuTorchSentenceTransformer:
    """ExecuTorch exported sentence transformer model (XNNPack or CPU backend)."""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the ExecuTorch model
        try:
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch,
            )

            logger.info(f"Loading ExecuTorch model from {model_path}...")
            self.model = _load_for_executorch(model_path)
            logger.info("✅ ExecuTorch model loaded successfully")
        except ImportError as e:
            logger.error("Failed to import ExecuTorch portable_lib")
            logger.error("Make sure ExecuTorch is installed: pip install executorch")
            raise
        except Exception as e:
            logger.error(f"Failed to load ExecuTorch model: {e}")
            raise

    def encode(self, text: str, max_length: int = 128) -> np.ndarray:
        """Encode a sentence to embedding using ExecuTorch model."""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Run inference
        outputs = self.model.forward((input_ids, attention_mask))

        # Extract embedding
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            embedding = outputs[0]
        else:
            embedding = outputs

        # Convert to numpy
        if hasattr(embedding, "numpy"):
            return embedding.numpy()
        elif isinstance(embedding, torch.Tensor):
            return embedding.detach().numpy()
        else:
            return np.array(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return np.linalg.norm(a - b)


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean absolute error between two vectors."""
    return np.mean(np.abs(a - b))


def compare_embeddings(
    original_embedding: np.ndarray,
    executorch_embedding: np.ndarray,
    text: str,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Compare two embeddings and return similarity metrics.

    Returns:
        (cosine_similarity, l2_distance, mean_absolute_error)
    """
    cos_sim = cosine_similarity(original_embedding, executorch_embedding)
    l2_dist = l2_distance(original_embedding, executorch_embedding)
    mae = mean_absolute_error(original_embedding, executorch_embedding)

    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        logger.info(f"{'-'*80}")
        logger.info(f"Original embedding shape:    {original_embedding.shape}")
        logger.info(f"ExecuTorch embedding shape:  {executorch_embedding.shape}")
        logger.info(f"{'-'*80}")
        logger.info(
            f"Cosine Similarity:  {cos_sim:.6f}  {'✅' if cos_sim > 0.99 else '⚠️' if cos_sim > 0.95 else '❌'}"
        )
        logger.info(
            f"L2 Distance:        {l2_dist:.6f}  {'✅' if l2_dist < 0.1 else '⚠️' if l2_dist < 0.5 else '❌'}"
        )
        logger.info(
            f"Mean Abs Error:     {mae:.6f}  {'✅' if mae < 0.01 else '⚠️' if mae < 0.05 else '❌'}"
        )

        # Show first few dimensions for inspection
        logger.info(f"{'-'*80}")
        logger.info(f"First 5 dimensions comparison:")
        logger.info(f"Original:   {original_embedding.flatten()[:5]}")
        logger.info(f"ExecuTorch: {executorch_embedding.flatten()[:5]}")
        logger.info(
            f"Difference: {(original_embedding - executorch_embedding).flatten()[:5]}"
        )

    return cos_sim, l2_dist, mae


def main():
    parser = argparse.ArgumentParser(
        description="Compare embeddings from ExecuTorch model with original transformers model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the exported .pte model file",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--sentences",
        nargs="+",
        default=[
            "This is an example sentence.",
            "Another example sentence for testing.",
        ],
        help="Sentences to encode and compare",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1

    logger.info(f"\n{'='*80}")
    logger.info("Sentence Transformer Embedding Comparison")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"ExecuTorch model: {args.model_path}")
    logger.info(f"Number of sentences: {len(args.sentences)}")
    logger.info(f"Max length: {args.max_length}")

    # Load models
    logger.info(f"\n{'='*80}")
    logger.info("Loading models...")
    logger.info(f"{'='*80}")

    logger.info("Loading original transformers model...")
    original_model = OriginalSentenceTransformer(args.model_name)
    logger.info("✅ Original model loaded")

    logger.info("Loading ExecuTorch model...")
    executorch_model = ExecuTorchSentenceTransformer(args.model_path, args.model_name)
    logger.info("✅ ExecuTorch model loaded")

    # Compare embeddings for each sentence
    results = []
    for i, sentence in enumerate(args.sentences, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing sentence {i}/{len(args.sentences)}")
        logger.info(f"{'='*80}")

        # Get embeddings
        logger.info("Generating embeddings...")
        original_emb = original_model.encode(sentence, args.max_length)
        executorch_emb = executorch_model.encode(sentence, args.max_length)

        # Compare
        cos_sim, l2_dist, mae = compare_embeddings(
            original_emb, executorch_emb, sentence, verbose=True
        )

        results.append(
            {
                "sentence": sentence,
                "cosine_similarity": cos_sim,
                "l2_distance": l2_dist,
                "mae": mae,
            }
        )

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")

    avg_cos_sim = np.mean([r["cosine_similarity"] for r in results])
    avg_l2_dist = np.mean([r["l2_distance"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])

    logger.info(f"Average Cosine Similarity: {avg_cos_sim:.6f}")
    logger.info(f"Average L2 Distance:       {avg_l2_dist:.6f}")
    logger.info(f"Average Mean Abs Error:    {avg_mae:.6f}")

    # Overall assessment
    logger.info(f"\n{'='*80}")
    if avg_cos_sim > 0.99 and avg_l2_dist < 0.1:
        logger.info(
            "✅ EXCELLENT: ExecuTorch model matches original model very closely!"
        )
    elif avg_cos_sim > 0.95 and avg_l2_dist < 0.5:
        logger.info(
            "⚠️  GOOD: ExecuTorch model is close to original model with minor differences"
        )
    else:
        logger.info("❌ WARNING: Significant differences detected between models")
        logger.info("   This may indicate an issue with the export process")

    logger.info(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
