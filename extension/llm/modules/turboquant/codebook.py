# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Lloyd-Max codebook solver and rotation matrix generator adapted from
# turboquant-vllm (Alberto-Codes/turboquant-vllm, Apache 2.0).
#
# Reference: arXiv 2504.19874 — "TurboQuant: Online Vector Quantization
# with Near-optimal Distortion Rate" (ICLR 2026).

"""
Lloyd-Max optimal scalar quantizer and random rotation matrix for TurboQuant.

After random orthogonal rotation, each coordinate of a unit-norm vector
follows a distribution concentrated near zero (Beta for exact, Gaussian
N(0, 1/d) for the d >= 64 approximation). The Lloyd-Max algorithm finds
the optimal centroids minimizing MSE for this distribution.

Results are cached so multi-layer models pay the scipy cost only once.
"""

import math
from functools import lru_cache

import torch


def solve_lloyd_max(
    d: int,
    bits: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal Lloyd-Max centroids and boundaries.

    Uses the Gaussian approximation N(0, 1/d) for the rotated coordinate
    distribution, which is accurate for d >= 64.

    Args:
        d: Vector dimension.
        bits: Quantization bits (produces 2^bits centroids).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        (centroids, boundaries) as 1-D float32 tensors.
        centroids has length 2^bits, boundaries has length 2^bits - 1.
    """
    return _solve_lloyd_max_cached(d, bits, max_iter, tol)


@lru_cache(maxsize=32)
def _solve_lloyd_max_cached(
    d: int, bits: int, max_iter: int, tol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy import integrate
        from scipy.stats import norm
    except ImportError:
        raise ImportError(
            "scipy is required for TurboQuant codebook computation. "
            "Install it with: pip install scipy"
        )

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.0 * sigma, 3.0 * sigma

    def pdf(x):
        return float(norm.pdf(x, loc=0.0, scale=sigma))

    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo] + boundaries + [hi]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            if b - a < 1e-15:
                new_centroids.append((a + b) / 2.0)
                continue
            numer, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denom, _ = integrate.quad(pdf, a, b)
            if denom < 1e-15:
                new_centroids.append((a + b) / 2.0)
            else:
                new_centroids.append(numer / denom)

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries_final = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries_final, dtype=torch.float32),
    )


def generate_rotation_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix via QR.

    Args:
        dim: Matrix dimension (d x d).
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (dim, dim) in float32 on CPU.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=gen, device="cpu", dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign.unsqueeze(0)
