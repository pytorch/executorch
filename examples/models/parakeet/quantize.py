"""Quantization utilities for Parakeet model export."""

from typing import Optional

import torch


def quantize_model_(  # noqa: C901
    module: torch.nn.Module,
    qlinear_config: Optional[str] = None,
    qlinear_group_size: int = 32,
    qlinear_packing_format: Optional[str] = None,
    qembedding_config: Optional[str] = None,
    qembedding_group_size: int = 0,
) -> None:
    """Quantize linear and embedding layers in a module in-place.

    Args:
        module: The PyTorch module to quantize.
        qlinear_config: Quantization config for linear layers ("4w", "8w", "8da4w", "8da8w").
        qlinear_group_size: Group size for linear quantization (default: 32).
        qlinear_packing_format: Packing format for linear layers (e.g., "tile_packed_to_4d").
        qembedding_config: Quantization config for embedding layers ("4w", "8w").
        qembedding_group_size: Group size for embedding quantization (default: 0 = per-axis).
    """
    if not qlinear_config and not qembedding_config:
        return

    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
        quantize_,
    )

    # Quantize embedding layers first
    if qembedding_config:
        if qembedding_group_size == 0:
            embedding_granularity = PerAxis(0)
        else:
            assert (
                qembedding_group_size % 2 == 0
            ), "Embedding group size must be a multiple of 2."
            embedding_granularity = PerGroup(qembedding_group_size)

        embedding_config = IntxWeightOnlyConfig(
            weight_dtype=torch.int4 if qembedding_config == "4w" else torch.int8,
            granularity=embedding_granularity,
        )

        print(
            f"  Applying {qembedding_config} embedding quantization "
            f"(group_size={qembedding_group_size})..."
        )
        quantize_(
            module,
            embedding_config,
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

    # Quantize linear layers
    if qlinear_config:
        # Determine granularity
        if qlinear_group_size == 0:
            granularity = PerAxis(0)
        else:
            granularity = PerGroup(qlinear_group_size)

        # Build quantization config
        if qlinear_config == "4w":
            if qlinear_packing_format:
                config = Int4WeightOnlyConfig(
                    group_size=qlinear_group_size,
                    int4_packing_format=qlinear_packing_format,
                    int4_choose_qparams_algorithm="hqq",
                )
            else:
                config = IntxWeightOnlyConfig(
                    weight_dtype=torch.int4,
                    granularity=granularity,
                )
        elif qlinear_config == "8w":
            config = IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=granularity,
            )
        elif qlinear_config == "8da4w":
            config = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=granularity,
            )
        elif qlinear_config == "8da8w":
            config = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8,
                weight_granularity=PerAxis(0),
            )
        else:
            raise ValueError(f"Unsupported qlinear_config: {qlinear_config}")

        # Filter: only quantize Linear layers with compatible dimensions
        def linear_filter(m, fqn):
            if isinstance(m, torch.nn.Linear):
                if qlinear_group_size == 0:
                    return True
                return m.weight.shape[1] % qlinear_group_size == 0
            return False

        print(
            f"  Applying {qlinear_config} linear quantization "
            f"(group_size={qlinear_group_size}, packing={qlinear_packing_format})..."
        )
        quantize_(module, config, filter_fn=linear_filter)
