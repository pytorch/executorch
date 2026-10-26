# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch

from executorch.backends.arm.vgf.quantization_quality import (
    compute_vgf_quantization_metrics,
)
from executorch.devtools.inspector.numerical_comparator.mse_numerical_comparator import (
    MSEComparator,
)
from executorch.devtools.inspector.numerical_comparator.snr_numerical_comparator import (
    SNRComparator,
)


def test_reuses_existing_executorch_mse_and_snr_comparators() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    quantized = torch.tensor([1.0, 2.0, 2.5])

    with mock.patch.object(
        MSEComparator,
        "element_compare",
        return_value=12.5,
    ) as mse_compare, mock.patch.object(
        SNRComparator,
        "element_compare",
        return_value=34.0,
    ) as snr_compare:
        metrics = compute_vgf_quantization_metrics(reference, quantized)

    assert metrics.mse == 12.5
    assert metrics.snr_db == 34.0
    mse_compare.assert_called_once()
    snr_compare.assert_called_once()


def test_reports_error_similarity_range_and_clipping_metrics() -> None:
    reference = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    quantized = torch.tensor([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])

    metrics = compute_vgf_quantization_metrics(
        reference,
        quantized,
        clip_min=-1.0,
        clip_max=1.0,
    )

    expected_mse = MSEComparator().element_compare(reference, quantized)
    expected_snr = SNRComparator().element_compare(reference, quantized)
    expected_cosine = torch.nn.functional.cosine_similarity(
        reference.double(),
        quantized.double(),
        dim=0,
    ).item()
    expected_relative_error = (
        torch.linalg.vector_norm((reference - quantized).double())
        / torch.linalg.vector_norm(reference.double())
    ).item()

    assert metrics.mse == pytest.approx(expected_mse)
    assert metrics.snr_db == pytest.approx(expected_snr)
    assert metrics.cosine_similarity == pytest.approx(expected_cosine)
    assert metrics.max_abs_error == pytest.approx(1.0)
    assert metrics.relative_error == pytest.approx(expected_relative_error)
    assert metrics.reference_min == pytest.approx(-2.0)
    assert metrics.reference_max == pytest.approx(2.0)
    assert metrics.quantized_min == pytest.approx(-1.0)
    assert metrics.quantized_max == pytest.approx(1.0)
    assert metrics.saturation_ratio == pytest.approx(4.0 / 7.0)
    assert metrics.clipping_ratio == pytest.approx(2.0 / 7.0)


def test_clipping_metrics_are_optional() -> None:
    metrics = compute_vgf_quantization_metrics(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 1.5]),
    )

    assert metrics.saturation_ratio is None
    assert metrics.clipping_ratio is None


def test_identical_tensors_have_zero_error_and_unit_cosine() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0])

    metrics = compute_vgf_quantization_metrics(reference, reference.clone())

    assert metrics.mse == pytest.approx(0.0)
    assert metrics.max_abs_error == pytest.approx(0.0)
    assert metrics.relative_error == pytest.approx(0.0)
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.snr_db == float("inf")


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_vgf_quantization_metrics(
            torch.ones(2),
            torch.ones(3),
        )


def test_rejects_empty_tensors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_vgf_quantization_metrics(
            torch.empty(0),
            torch.empty(0),
        )


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_vgf_quantization_metrics(
            torch.tensor([1.0, float("nan")]),
            torch.tensor([1.0, 2.0]),
        )


def test_rejects_incomplete_or_invalid_clipping_bounds() -> None:
    reference = torch.ones(2)
    quantized = torch.ones(2)

    with pytest.raises(ValueError, match="both be provided"):
        compute_vgf_quantization_metrics(
            reference,
            quantized,
            clip_min=-1.0,
        )

    with pytest.raises(ValueError, match="clip_min <= clip_max"):
        compute_vgf_quantization_metrics(
            reference,
            quantized,
            clip_min=1.0,
            clip_max=-1.0,
        )
