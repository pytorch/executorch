# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.extension.llm.batching.sampler import (
    BatchArgmax,
    BatchSampler,
    NUM_PARAMS,
    sample_rows,
)


def host_sample(logits, temperature, top_p, top_k, coin):
    """extension/llm/sampler/sampler.cpp's Sampler::sample, for one row."""
    vocab = logits.numel()
    logits = logits.double()
    if temperature == 0.0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperature, dim=-1)
    if 0 < top_k < vocab:
        order = torch.sort(probs, descending=True, stable=True).indices[:top_k]
        r = coin * float(probs[order].sum())
        cdf = 0.0
        for index in order.tolist():
            cdf += float(probs[index])
            if r < cdf:
                return index
        return int(order[-1])
    if top_p <= 0 or top_p >= 1:
        cdf = 0.0
        for index in range(vocab):
            cdf += float(probs[index])
            if coin < cdf:
                return index
        return vocab - 1
    order = torch.sort(probs, descending=True, stable=True).indices
    cumulative = 0.0
    last = len(order) - 1
    for i, index in enumerate(order.tolist()):
        cumulative += float(probs[index])
        if cumulative > top_p:
            last = i
            break
    r = coin * cumulative
    cdf = 0.0
    for index in order[: last + 1].tolist():
        cdf += float(probs[index])
        if r < cdf:
            return index
    return int(order[last])


def params_of(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, NUM_PARAMS)


class SampleRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.logits = torch.randn(6, 50) * 3

    def check(self, temperature, top_p, top_k, coins):
        for coin in coins:
            params = params_of(
                [[temperature, top_p, top_k, coin]] * self.logits.shape[0]
            )
            got = sample_rows(self.logits, params).tolist()
            want = [
                host_sample(row, temperature, top_p, top_k, coin) for row in self.logits
            ]
            self.assertEqual(got, want, (temperature, top_p, top_k, coin))

    def test_temperature_zero_is_argmax(self) -> None:
        self.check(0.0, 0.9, 5, [0.0, 0.5, 0.999])

    def test_full_distribution_samples_in_token_order(self) -> None:
        self.check(0.8, 1.0, 0, [0.0, 0.1, 0.37, 0.5, 0.92, 0.9999])

    def test_top_k(self) -> None:
        self.check(1.2, 1.0, 4, [0.0, 0.2, 0.55, 0.8, 0.9999])

    def test_top_k_takes_precedence_over_top_p(self) -> None:
        self.check(1.0, 0.3, 7, [0.05, 0.6, 0.97])

    def test_top_p(self) -> None:
        self.check(0.7, 0.85, 0, [0.0, 0.25, 0.5, 0.75, 0.9999])
        self.check(1.5, 0.2, 0, [0.0, 0.5, 0.9999])

    def test_each_row_has_its_own_policy(self) -> None:
        rows = [
            [0.0, 1.0, 0, 0.3],
            [0.9, 1.0, 0, 0.3],
            [0.9, 1.0, 3, 0.7],
            [0.6, 0.8, 0, 0.1],
            [0.0, 0.5, 8, 0.9],
            [1.3, 0.95, 0, 0.99],
        ]
        got = sample_rows(self.logits, params_of(rows)).tolist()
        want = [
            host_sample(logits, t, p, int(k), c)
            for logits, (t, p, k, c) in zip(self.logits, rows)
        ]
        self.assertEqual(got, want)

    def test_matches_empirical_distribution(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
        coins = torch.rand(4000)
        params = torch.stack([torch.tensor([1.0, 1.0, 0.0, float(c)]) for c in coins])
        tokens = sample_rows(logits.expand(len(coins), -1), params)
        frequencies = torch.bincount(tokens, minlength=4).double() / len(coins)
        expected = torch.softmax(logits[0].double(), dim=-1)
        self.assertLess(float((frequencies - expected).abs().max()), 0.03)

    def test_bfloat16_logits(self) -> None:
        params = params_of([[0.0, 1.0, 0, 0.5]] * 6)
        self.assertEqual(
            sample_rows(self.logits.bfloat16(), params).tolist(),
            torch.argmax(self.logits.bfloat16().float(), dim=-1).tolist(),
        )


class ExportTest(unittest.TestCase):
    def test_modules_export_with_dynamic_rows(self) -> None:
        rows = torch.export.Dim("rows", min=1, max=64)
        logits = torch.randn(8, 50)
        params = params_of([[0.7, 0.9, 0, 0.4]] * 8)
        sampler = torch.export.export(
            BatchSampler(),
            (logits, params),
            dynamic_shapes=({0: rows}, {0: rows}),
            strict=True,
        )
        argmax = torch.export.export(
            BatchArgmax(), (logits,), dynamic_shapes=({0: rows},), strict=True
        )
        for count in (1, 3, 8):
            self.assertEqual(
                sampler.module()(logits[:count], params[:count]).tolist(),
                sample_rows(logits[:count], params[:count]).tolist(),
            )
            self.assertEqual(
                argmax.module()(logits[:count]).tolist(),
                torch.argmax(logits[:count], dim=-1).tolist(),
            )


if __name__ == "__main__":
    unittest.main()
