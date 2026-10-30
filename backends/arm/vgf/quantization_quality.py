# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Quantization-quality metrics for VGF tensor comparisons.

The generic ExecuTorch numerical-comparator package already implements MSE and
SNR. This module reuses those implementations and only aggregates the additional
metrics useful when inspecting VGF quantization quality.

The reference and quantized tensors must be in the same numerical domain. For
integer affine quantization, compare the floating-point reference against the
dequantized tensor, and pass the dequantized clipping bounds when saturation and
clipping ratios are required.

``relative_error`` is the L2 error norm divided by the L2 reference norm.
``saturation_ratio`` is the fraction of candidate values at either clipping bound,
and ``clipping_ratio`` is the fraction of reference values outside those bounds.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from executorch.devtools.inspector.numerical_comparator.mse_numerical_comparator import (
    MSEComparator,
)
from executorch.devtools.inspector.numerical_comparator.snr_numerical_comparator import (
    SNRComparator,
)


@dataclass(frozen=True)
class VgfQuantizationMetrics:
    """Aggregated quality metrics for one reference/quantized tensor pair."""

    mse: float
    snr_db: float
    cosine_similarity: float
    max_abs_error: float
    relative_error: float
    reference_min: float
    reference_max: float
    quantized_min: float
    quantized_max: float
    saturation_ratio: float | None
    clipping_ratio: float | None


def _prepare_tensors(
    reference: torch.Tensor,
    quantized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape != quantized.shape:
        raise ValueError(
            "reference and quantized tensors must have the same shape; "
            f"got {tuple(reference.shape)} and {tuple(quantized.shape)}"
        )
    if reference.numel() == 0:
        raise ValueError("reference and quantized tensors must be non-empty")

    reference_f = reference.detach().cpu().to(torch.float64)
    quantized_f = quantized.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(reference_f).all()) or not bool(
        torch.isfinite(quantized_f).all()
    ):
        raise ValueError("reference and quantized tensors must contain finite values")

    return reference_f, quantized_f


def _validate_clipping_bounds(
    clip_min: float | None,
    clip_max: float | None,
) -> None:
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must both be provided or both be None")
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clipping bounds must satisfy clip_min <= clip_max")


def compute_vgf_quantization_metrics(
    reference: torch.Tensor,
    quantized: torch.Tensor,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
    eps: float = 1.0e-12,
) -> VgfQuantizationMetrics:
    """Compute VGF quantization-quality metrics for a tensor pair.

    Args:
        reference: Floating-point reference tensor.
        quantized: Quantized result expressed in the same numerical domain as
            ``reference``. For affine integer quantization, this is normally the
            dequantized tensor.
        clip_min: Optional lower representable value in the same numerical domain.
        clip_max: Optional upper representable value in the same numerical domain.
        eps: Positive epsilon used by cosine similarity and relative error.

    Returns:
        Aggregated VGF quantization-quality metrics.

    Raises:
        ValueError: If the tensors are empty, have different shapes, contain
            non-finite values, or clipping arguments are invalid.

    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    _validate_clipping_bounds(clip_min, clip_max)
    reference_f, quantized_f = _prepare_tensors(reference, quantized)

    # Reuse the generic ExecuTorch comparators instead of maintaining VGF copies.
    mse = MSEComparator().element_compare(reference_f, quantized_f)
    snr_db = SNRComparator().element_compare(reference_f, quantized_f)

    flat_reference = reference_f.flatten()
    flat_quantized = quantized_f.flatten()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        flat_reference,
        flat_quantized,
        dim=0,
        eps=eps,
    ).item()

    abs_error = torch.abs(reference_f - quantized_f)
    max_abs_error = abs_error.max().item()

    error_norm = torch.linalg.vector_norm(reference_f - quantized_f).item()
    reference_norm = torch.linalg.vector_norm(reference_f).item()
    relative_error = error_norm / max(reference_norm, eps)

    saturation_ratio: float | None = None
    clipping_ratio: float | None = None
    if clip_min is not None and clip_max is not None:
        at_lower_bound = torch.isclose(
            quantized_f,
            torch.tensor(clip_min, dtype=torch.float64),
            rtol=0.0,
            atol=eps,
        )
        at_upper_bound = torch.isclose(
            quantized_f,
            torch.tensor(clip_max, dtype=torch.float64),
            rtol=0.0,
            atol=eps,
        )
        saturation_ratio = torch.mean(
            (at_lower_bound | at_upper_bound).to(torch.float64)
        ).item()
        clipping_ratio = torch.mean(
            ((reference_f < clip_min) | (reference_f > clip_max)).to(torch.float64)
        ).item()

    return VgfQuantizationMetrics(
        mse=mse,
        snr_db=snr_db,
        cosine_similarity=cosine_similarity,
        max_abs_error=max_abs_error,
        relative_error=relative_error,
        reference_min=reference_f.min().item(),
        reference_max=reference_f.max().item(),
        quantized_min=quantized_f.min().item(),
        quantized_max=quantized_f.max().item(),
        saturation_ratio=saturation_ratio,
        clipping_ratio=clipping_ratio,
    )
