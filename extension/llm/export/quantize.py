"""TorchAO source-transform quantization for ExecuTorch LLM export.

Applies quantization to nn.Linear and nn.Embedding layers in-place before
torch.export(). This is the source-transform counterpart to quantizer_lib.py
(which handles PT2E graph-mode quantization).

Supported linear configs: "4w", "8w", "8da4w", "8da8w", "fpa4w", "nvfp4".
Supported embedding configs: "4w", "8w", "nvfp4".

Usage:
    from executorch.extension.llm.export.quantize import quantize_model_
    quantize_model_(model, qlinear_config="8da4w", qembedding_config="4w")
"""

from typing import Optional

import torch
from executorch.exir._warnings import experimental


def _make_granularity(group_size: int):
    """Create PerAxis(0) or PerGroup granularity."""
    from torchao.quantization.granularity import PerAxis, PerGroup

    return PerAxis(0) if group_size == 0 else PerGroup(group_size)


def _make_linear_config(config_name: str, group_size: int, packing_format=None):
    """Build a TorchAO config for linear layer quantization."""
    from torchao.quantization.quant_api import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
    )

    granularity = _make_granularity(group_size)

    if config_name == "nvfp4":
        from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Config

        assert group_size == 16, "NVFP4 requires group_size=16"
        return ExportableNVFP4Config(use_per_tensor_scale=False)
    elif config_name == "4w":
        if packing_format:
            return Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format=packing_format,
                int4_choose_qparams_algorithm="hqq",
            )
        return IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    elif config_name == "8w":
        return IntxWeightOnlyConfig(
            weight_dtype=torch.int8,
            granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    elif config_name == "8da4w":
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    elif config_name == "8da8w":
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8,
            weight_granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    else:
        raise ValueError(f"Unsupported qlinear_config: {config_name}")


def _make_embedding_config(config_name: str, group_size: int):
    """Build a TorchAO config for embedding layer quantization."""
    from torchao.quantization.quant_api import IntxWeightOnlyConfig

    if group_size != 0:
        assert group_size % 2 == 0, "Embedding group size must be a multiple of 2."

    granularity = _make_granularity(group_size)

    if config_name == "nvfp4":
        from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Config

        return ExportableNVFP4Config(use_per_tensor_scale=False)
    elif config_name == "4w":
        return IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    elif config_name == "8w":
        return IntxWeightOnlyConfig(
            weight_dtype=torch.int8,
            granularity=granularity,
            intx_choose_qparams_algorithm="hqq_scale_only",
        )
    else:
        raise ValueError(f"Unsupported qembedding_config: {config_name}")


def _check_shape_compatible(m, fqn, config_name, group_size, skip_incompatible_shapes):
    """Check shape compatibility. Returns True if compatible, False if skipped.

    Raises RuntimeError if incompatible and skip_incompatible_shapes is False.
    """
    shape = m.weight.shape
    if config_name == "nvfp4":
        compatible = shape[-2] % group_size == 0 and shape[-1] % group_size == 0
    elif config_name == "fpa4w":
        # MPS UIntx kernel requires N % 4 == 0 when M > 1 (e.g. prefill)
        compatible = shape[-1] % group_size == 0 and shape[-2] % 4 == 0
    elif group_size != 0:
        compatible = shape[-1] % group_size == 0
    else:
        compatible = True

    if compatible:
        return True
    if not skip_incompatible_shapes:
        raise RuntimeError(
            f"Layer {fqn} has weight shape {shape} "
            f"incompatible with {config_name} (group_size={group_size}). "
            f"Use skip_incompatible_shapes=True to skip instead of failing."
        )
    print(
        f"  Skipping {fqn}: weight shape {shape} "
        f"incompatible with {config_name} (group_size={group_size})"
    )
    return False


def _make_linear_filter(
    config_name: str, group_size: int, skip_incompatible_shapes: bool = False
):
    """Create a filter_fn for linear layers, skipping incompatible shapes."""

    def linear_filter(m, fqn):
        if not isinstance(m, torch.nn.Linear):
            return False
        return _check_shape_compatible(
            m, fqn, config_name, group_size, skip_incompatible_shapes
        )

    return linear_filter


def _make_embedding_filter(
    config_name: str, group_size: int, skip_incompatible_shapes: bool = False
):
    """Create a filter_fn for embedding layers, skipping incompatible shapes."""

    def embedding_filter(m, fqn):
        if not isinstance(m, torch.nn.Embedding):
            return False
        return _check_shape_compatible(
            m, fqn, config_name, group_size, skip_incompatible_shapes
        )

    return embedding_filter


def _default_group_size(config_name: Optional[str]) -> int:
    """Return the default group size for a quantization config."""
    if config_name == "nvfp4":
        return 16
    if config_name in ("8w", "8da8w"):
        return 0
    return 32


@experimental("quantize_model_ is experimental and may change without notice.")
def quantize_model_(
    module: torch.nn.Module,
    qlinear_config: Optional[str] = None,
    qlinear_group_size: Optional[int] = None,
    qlinear_packing_format: Optional[str] = None,
    qembedding_config: Optional[str] = None,
    qembedding_group_size: Optional[int] = None,
    tie_word_embeddings: bool = False,
    skip_incompatible_shapes: bool = False,
) -> None:
    """Quantize linear and embedding layers in a module in-place.

    .. warning::

        This API is experimental and may change without notice.

    Args:
        module: The PyTorch module to quantize.
        qlinear_config: Quantization config for linear layers
            ("4w", "8w", "8da4w", "8da8w", "fpa4w", "nvfp4").
        qlinear_group_size: Group size for linear quantization.
            Defaults to 16 for nvfp4, 32 for 4w/8da4w, 0 (per-axis) for 8w/8da8w.
        qlinear_packing_format: Packing format for linear layers
            (e.g., "tile_packed_to_4d").
        qembedding_config: Quantization config for embedding layers
            ("4w", "8w", "nvfp4").
        qembedding_group_size: Group size for embedding quantization.
            Defaults to 16 for nvfp4, 32 for 4w, 0 (per-axis) for 8w.
        tie_word_embeddings: If True and both linear and embedding use the
            same quantization, re-tie lm_head.weight to embed_tokens.weight
            after quantization.
        skip_incompatible_shapes: If True, silently skip layers with
            incompatible weight shapes. If False (default), raise RuntimeError.
    """
    if not qlinear_config and not qembedding_config:
        return

    if qlinear_group_size is None:
        qlinear_group_size = _default_group_size(qlinear_config)
    if qembedding_group_size is None:
        qembedding_group_size = _default_group_size(qembedding_config)

    from torchao.quantization.quant_api import quantize_

    # Metal (MPS) quantization uses a separate API
    if qlinear_config == "fpa4w":
        import torchao.experimental.ops.mps  # noqa: F401
        from torchao.experimental.quant_api import UIntxWeightOnlyConfig

        config = UIntxWeightOnlyConfig(
            group_size=qlinear_group_size,
            bitwidth=4,
            uintx_choose_qparams_algorithm="hqq",
        )
        print(
            f"  Applying {qlinear_config} linear quantization "
            f"(group_size={qlinear_group_size})..."
        )
        quantize_(
            module,
            config,
            filter_fn=_make_linear_filter(
                "fpa4w", qlinear_group_size, skip_incompatible_shapes
            ),
        )
        return

    # Quantize embedding layers first
    if qembedding_config:
        config = _make_embedding_config(qembedding_config, qembedding_group_size)
        print(
            f"  Applying {qembedding_config} embedding quantization "
            f"(group_size={qembedding_group_size})..."
        )
        quantize_(
            module,
            config,
            filter_fn=_make_embedding_filter(
                qembedding_config, qembedding_group_size, skip_incompatible_shapes
            ),
        )

    # Quantize linear layers
    if qlinear_config:
        config = _make_linear_config(
            qlinear_config, qlinear_group_size, qlinear_packing_format
        )
        print(
            f"  Applying {qlinear_config} linear quantization "
            f"(group_size={qlinear_group_size}, packing={qlinear_packing_format})..."
        )
        quantize_(
            module,
            config,
            filter_fn=_make_linear_filter(
                qlinear_config, qlinear_group_size, skip_incompatible_shapes
            ),
        )

    # Re-tie word embeddings after quantization if both use the same config
    if (
        tie_word_embeddings
        and qlinear_config == qembedding_config
        and hasattr(module, "lm_head")
        and hasattr(module, "model")
    ):
        embed = getattr(module.model, "embed_tokens", None)
        if embed is not None:
            module.lm_head.weight = embed.weight
            print("  Re-tied lm_head weights to embedding (tie_word_embeddings=True)")
