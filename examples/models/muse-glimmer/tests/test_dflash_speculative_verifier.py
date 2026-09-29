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
    DFlashSpeculativeVerifier,
    verify_speculative,
)


def _probabilities(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        token = max(range(len(logits)), key=lambda index: (logits[index], -index))
        return [float(index == token) for index in range(len(logits))]
    maximum = max(logits)
    weights = [math.exp((logit - maximum) / temperature) for logit in logits]
    return [weight / sum(weights) for weight in weights]


def _categorical(probabilities: list[float], uniform: float) -> int:
    cumulative = 0.0
    for token, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= uniform:
            return token
    return len(probabilities) - 1


def _strict_categorical(probabilities: list[float], uniform: float) -> int:
    cumulative, last_supported = 0.0, 0
    for token, probability in enumerate(probabilities):
        if probability == 0.0:
            continue
        last_supported = token
        cumulative += probability
        if uniform < cumulative:
            return token
    return last_supported


def _reference_verify(
    target_logits: torch.Tensor,
    draft_probabilities: torch.Tensor,
    candidates: list[int],
    temperature: float,
    draft_argmax: bool,
    accept_uniforms: list[float],
    correction_uniform: float,
) -> tuple[int, int]:
    """Mirrors the host verifier loop in dflash_session.cpp."""
    target = [_probabilities(row.tolist(), temperature) for row in target_logits]
    draft = draft_probabilities.tolist()
    for row in range(len(candidates) - 1):
        token = candidates[row + 1]
        if temperature <= 0.0:
            pick = max(
                range(target_logits.shape[1]),
                key=lambda index: (float(target_logits[row, index]), -index),
            )
            if pick != token:
                return row + 1, pick
            continue
        p = target[row][token]
        q = 1.0 if draft_argmax else draft[row][token]
        if accept_uniforms[row] < (min(1.0, p / q) if q > 0.0 else 1.0):
            continue
        if draft_argmax:
            excluded = list(target[row])
            excluded[token] = 0.0
            total = sum(excluded)
            return row + 1, _strict_categorical(
                [value / total for value in excluded], correction_uniform
            )
        residual = [max(0.0, t - d) for t, d in zip(target[row], draft[row])]
        total = sum(residual)
        correction = [v / total for v in residual] if total > 0.0 else target[row]
        return row + 1, _categorical(correction, correction_uniform)

    final = target[-1]
    if temperature <= 0.0:
        return len(candidates), max(
            range(len(final)), key=lambda index: (final[index], -index)
        )
    return len(candidates), _categorical(final, correction_uniform)


class DFlashSpeculativeVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [8, 0, 1, 2]
        self.draft_probabilities = torch.tensor(
            [[0.6, 0.3, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1], [0.2, 0.2, 0.5, 0.1]]
        )
        self.target_probabilities = torch.tensor(
            [
                [0.5, 0.2, 0.2, 0.1],
                [0.1, 0.3, 0.4, 0.2],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        self.target_logits = torch.log(self.target_probabilities)

    def _check(
        self,
        *,
        temperature: float,
        draft_argmax: bool,
        accept_uniforms: list[float],
        correction_uniform: float,
        target_logits: torch.Tensor | None = None,
        draft_probabilities: torch.Tensor | None = None,
        num_proposals: int = 3,
    ) -> None:
        target_logits = (
            self.target_logits if target_logits is None else target_logits
        )[: num_proposals + 1]
        draft_probabilities = (
            self.draft_probabilities
            if draft_probabilities is None
            else draft_probabilities
        )[:num_proposals]
        candidates = self.candidates[: num_proposals + 1]
        actual = verify_speculative(
            target_logits,
            draft_probabilities,
            torch.tensor(candidates),
            torch.tensor([temperature]),
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor([draft_argmax]),
            torch.tensor(accept_uniforms[:num_proposals]),
            torch.tensor([correction_uniform]),
        )
        committed, correction = _reference_verify(
            target_logits,
            draft_probabilities,
            candidates,
            temperature,
            draft_argmax,
            accept_uniforms,
            correction_uniform,
        )
        self.assertEqual(actual.tolist(), [committed, correction, *candidates])

    def test_greedy_rejection_at_every_row(self) -> None:
        for rejection_row in range(3):
            logits = torch.full((4, 4), -10.0)
            for row in range(4):
                logits[row, self.candidates[min(row + 1, 3)]] = 10.0
            logits[rejection_row] = torch.tensor([0.0, 1.0, 2.0, 3.0])
            with self.subTest(rejection_row=rejection_row):
                self._check(
                    temperature=0.0,
                    draft_argmax=False,
                    accept_uniforms=[0.0] * 3,
                    correction_uniform=0.0,
                    target_logits=logits,
                )

    def test_all_accepted_uses_final_target_row(self) -> None:
        for temperature in (0.0, 1.0):
            with self.subTest(temperature=temperature):
                self._check(
                    temperature=temperature,
                    draft_argmax=False,
                    accept_uniforms=[0.0] * 3,
                    correction_uniform=0.75,
                    target_logits=(
                        torch.log(torch.eye(4) * 0.9 + 0.025)
                        if temperature == 0.0
                        else None
                    ),
                )

    def test_stochastic_residual_rejection_at_every_row(self) -> None:
        for rejection_row in range(3):
            uniforms = [0.0] * 3
            uniforms[rejection_row] = 0.99
            with self.subTest(rejection_row=rejection_row):
                self._check(
                    temperature=1.0,
                    draft_argmax=False,
                    accept_uniforms=uniforms,
                    correction_uniform=0.4,
                )

    def test_draft_argmax_uses_strict_excluded_token_cdf(self) -> None:
        logits = self.target_logits.clone()
        logits[0] = torch.tensor([0.0, 0.0, 0.0, float("-inf")])
        self._check(
            temperature=1.0,
            draft_argmax=True,
            accept_uniforms=[0.75, 0.0, 0.0],
            correction_uniform=0.5,
            target_logits=logits,
        )

    def test_zero_draft_probability_is_always_accepted(self) -> None:
        draft = self.draft_probabilities.clone()
        draft[0, self.candidates[1]] = 0.0
        self._check(
            temperature=1.0,
            draft_argmax=False,
            accept_uniforms=[0.999, 0.0, 0.0],
            correction_uniform=0.55,
            draft_probabilities=draft,
        )

    def test_zero_residual_falls_back_to_target_distribution(self) -> None:
        self._check(
            temperature=1.0,
            draft_argmax=False,
            accept_uniforms=[1.0, 0.0, 0.0],
            correction_uniform=0.5,
            draft_probabilities=self.target_probabilities[:3].clone(),
        )

    def test_fewer_proposals_than_exported_maximum(self) -> None:
        for num_proposals in (1, 2):
            with self.subTest(num_proposals=num_proposals):
                self._check(
                    temperature=1.0,
                    draft_argmax=False,
                    accept_uniforms=[0.0, 0.99, 0.0],
                    correction_uniform=0.25,
                    num_proposals=num_proposals,
                )

    def test_strict_export_supports_dynamic_proposal_count(self) -> None:
        proposals = torch.export.Dim("proposals", min=1, max=3)
        inputs = (
            self.target_logits,
            self.draft_probabilities,
            torch.tensor(self.candidates),
            torch.tensor([0.0]),
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor([False]),
        )
        exported = torch.export.export(
            DFlashSpeculativeVerifier(),
            inputs,
            dynamic_shapes=(
                {0: proposals + 1},
                {0: proposals},
                {0: proposals + 1},
                None,
                None,
                None,
                None,
            ),
            strict=True,
        ).module()
        for count in (1, 3):
            actual = exported(
                self.target_logits[: count + 1],
                self.draft_probabilities[:count],
                torch.tensor(self.candidates[: count + 1]),
                *inputs[3:],
            )
            expected = DFlashSpeculativeVerifier()(
                self.target_logits[: count + 1],
                self.draft_probabilities[:count],
                torch.tensor(self.candidates[: count + 1]),
                *inputs[3:],
            )
            self.assertEqual(actual.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
