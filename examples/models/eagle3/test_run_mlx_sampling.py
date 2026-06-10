# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributional test for the MLX runner's modified rejection sampling.

One speculative-sampling step (propose from the draft ``q``, then accept/reject
against the target ``p`` with a residual resample over the full target vocab)
must reproduce ``p`` exactly, including when the draft's reduced vocab covers
only some target ids. Monte-Carlo checks the implemented ``_accept`` path.
"""

import torch

from executorch.examples.models.eagle3.run_mlx import _accept


def test_rejection_sampling_reproduces_target_distribution():
    gen = torch.Generator().manual_seed(0)

    V, Vd = 8, 5
    # Draft reaches target ids [0, 3, 4, 6, 7]; {1, 2, 5} are draft-unreachable.
    d2t = torch.tensor([0, 2, 2, 3, 3], dtype=torch.long)
    target_ids_all = torch.arange(Vd, dtype=torch.long) + d2t
    assert target_ids_all.tolist() == [0, 3, 4, 6, 7]

    p = torch.softmax(torch.randn(V, generator=gen), dim=-1)
    q = torch.softmax(torch.randn(Vd, generator=gen), dim=-1)
    p_logits = p.log()  # softmax(log p) == p at temp 1

    n = 60000
    counts = torch.zeros(V)
    for _ in range(n):
        d = int(torch.multinomial(q, 1, generator=gen))
        x0 = int(target_ids_all[d])
        accepted, fallback = _accept(p_logits, x0, d, q, 1.0, target_ids_all, gen)
        counts[x0 if accepted else fallback] += 1

    empirical = counts / n
    assert torch.max(torch.abs(empirical - p)) < 0.02, (
        f"empirical={empirical.tolist()} target={p.tolist()}"
    )


def test_greedy_accept_is_exact_match():
    target_ids_all = torch.arange(4, dtype=torch.long)
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])  # argmax == 1
    accepted, corrected = _accept(logits, 1, 0, None, 0.0, target_ids_all, None)
    assert accepted and corrected == 1
    accepted, corrected = _accept(logits, 2, 0, None, 0.0, target_ids_all, None)
    assert (not accepted) and corrected == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
