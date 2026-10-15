# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import unittest

import torch

from executorch.examples.models.muse_glimmer.model.dflash_token_sampler import (
    DFlashTokenSampler,
    sample_tokens,
)


def _reference_row(
    logits: list[float], temperature: float, top_k: int, top_p: float, uniform: float
) -> tuple[int, list[float]]:
    vocab_size = len(logits)
    if temperature <= 0.0:
        token = max(range(vocab_size), key=lambda index: (logits[index], -index))
        return token, [float(index == token) for index in range(vocab_size)]

    ranked = sorted(range(vocab_size), key=lambda index: (-logits[index], index))
    if 0 < top_k < vocab_size:
        ranked = ranked[:top_k]
    maximum = max(logits[index] for index in ranked)
    weights = [0.0] * vocab_size
    for index in ranked:
        weights[index] = math.exp((logits[index] - maximum) / temperature)
    total = sum(weights)
    if 0.0 < top_p < 1.0:
        nucleus, nucleus_sum = [], 0.0
        for index in ranked:
            if nucleus_sum >= top_p * total:
                break
            nucleus.append(index)
            nucleus_sum += weights[index]
        weights = [w if i in nucleus else 0.0 for i, w in enumerate(weights)]
        total = nucleus_sum

    probabilities = [weight / total for weight in weights]
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= uniform:
            return index, probabilities
    return vocab_size - 1, probabilities


def _scalar(value: float | int) -> torch.Tensor:
    dtype = torch.int64 if isinstance(value, int) else torch.float32
    return torch.tensor([value], dtype=dtype)


class DFlashTokenSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [
                [1.0, 3.0, 2.0, -1.0, 0.0],
                [2.0, 2.0, 0.0, 1.0, -2.0],
                [-1.0, 0.0, 4.0, 2.0, 1.0],
            ]
        )
        self.uniforms = torch.tensor([0.0, 0.5, 0.9])

    def test_matches_cpu_reference(self) -> None:
        for temperature, top_k, top_p in (
            (0.0, 0, 1.0),
            (1.0, 0, 1.0),
            (0.7, 2, 1.0),
            (1.3, 0, 0.6),
            (0.8, 3, 0.75),
        ):
            with self.subTest(temperature=temperature, top_k=top_k, top_p=top_p):
                tokens, probabilities = sample_tokens(
                    self.logits,
                    _scalar(temperature),
                    _scalar(top_k),
                    _scalar(top_p),
                    self.uniforms,
                )
                expected = [
                    _reference_row(row.tolist(), temperature, top_k, top_p, float(u))
                    for row, u in zip(self.logits, self.uniforms)
                ]
                self.assertEqual(tokens.tolist(), [token for token, _ in expected])
                torch.testing.assert_close(
                    probabilities,
                    torch.tensor([row for _, row in expected]),
                    atol=2e-5,
                    rtol=2e-5,
                )

    def test_stable_top_k_prefers_lowest_token_id(self) -> None:
        tokens, probabilities = sample_tokens(
            self.logits[1:2], _scalar(1.0), _scalar(1), _scalar(1.0), self.uniforms[:1]
        )
        self.assertEqual(tokens.tolist(), [0])
        self.assertEqual(probabilities.tolist(), [[1.0, 0.0, 0.0, 0.0, 0.0]])

    def test_inverse_cdf_uses_original_order_and_inclusive_boundary(self) -> None:
        tokens, _ = sample_tokens(
            torch.log(torch.tensor([[0.25, 0.75]])),
            _scalar(1.0),
            _scalar(0),
            _scalar(1.0),
            torch.tensor([0.25]),
        )
        self.assertEqual(tokens.tolist(), [0])

    def test_module_is_exact_argmax_at_zero_temperature(self) -> None:
        tied = torch.tensor([[3.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
        for _ in range(5):
            tokens, _ = DFlashTokenSampler()(
                tied, _scalar(0.0), _scalar(0), _scalar(1.0)
            )
            self.assertEqual(tokens.tolist(), [0, 1])

    def test_strict_export_supports_dynamic_rows(self) -> None:
        rows = torch.export.Dim("rows", min=1, max=3)
        exported = torch.export.export(
            DFlashTokenSampler(),
            (self.logits, _scalar(0.0), _scalar(0), _scalar(1.0)),
            dynamic_shapes=({0: rows}, None, None, None),
            strict=True,
        ).module()
        for count in (1, 3):
            tokens, probabilities = exported(
                self.logits[:count], _scalar(0.0), _scalar(0), _scalar(1.0)
            )
            self.assertEqual(tokens.tolist(), self.logits[:count].argmax(-1).tolist())
            self.assertEqual(probabilities.shape, torch.Size([count, 5]))


if __name__ == "__main__":
    unittest.main()
