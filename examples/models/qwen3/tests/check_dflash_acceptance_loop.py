# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic tests for the DFlash acceptance loop's core logic (review point
6): first_mismatch(), compute_block_len(), and truncate_at_eos(), imported
directly from run_dflash.py -- no model or hardware needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, "backends/mlx/examples/llm")
from run_dflash import compute_block_len, first_mismatch, truncate_at_eos


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    if not condition:
        raise AssertionError(name)


def test_first_token_rejected():
    draft_ids = [5, 6, 7]
    target_ids = [9, 100, 101, 102]
    accepted = first_mismatch(draft_ids, target_ids)
    check("first-token rejection: accepted == 0", accepted == 0)
    new_tokens = draft_ids[:accepted] + [target_ids[accepted]]
    check("first-token rejection: emits only target's token", new_tokens == [9])


def test_full_acceptance_with_bonus():
    draft_ids = [5, 6, 7]
    target_ids = [5, 6, 7, 42]
    accepted = first_mismatch(draft_ids, target_ids)
    check("full acceptance: accepted == len(draft_ids)", accepted == len(draft_ids))
    new_tokens = draft_ids[:accepted] + [target_ids[accepted]]
    check(
        "full acceptance: emits all draft tokens plus the bonus",
        new_tokens == [5, 6, 7, 42],
    )


def test_partial_acceptance():
    draft_ids = [5, 6, 7, 8]
    target_ids = [5, 6, 99, 100, 101]
    accepted = first_mismatch(draft_ids, target_ids)
    check("partial acceptance: accepted == 2", accepted == 2)
    new_tokens = draft_ids[:accepted] + [target_ids[accepted]]
    check(
        "partial acceptance: emits matched prefix plus target's replacement",
        new_tokens == [5, 6, 99],
    )


def test_eos_mid_block_truncates():
    EOS = 999
    new_tokens = [5, 6, EOS, 8, 9]
    accepted = 4
    truncated, clamped_accepted = truncate_at_eos(new_tokens, accepted, EOS)
    check("EOS mid-block: truncates right after EOS", truncated == [5, 6, EOS])
    check(
        "EOS mid-block: accepted is clamped to len(truncated) - 1",
        clamped_accepted == 2,
    )


def test_eos_not_present_is_a_no_op():
    EOS = 999
    new_tokens = [5, 6, 7]
    accepted = 2
    truncated, kept_accepted = truncate_at_eos(new_tokens, accepted, EOS)
    check("no EOS: new_tokens unchanged", truncated == new_tokens)
    check("no EOS: accepted unchanged", kept_accepted == accepted)


def test_eos_at_final_position():
    EOS = 999
    new_tokens = [5, 6, 7, EOS]
    accepted = 3
    truncated, kept_accepted = truncate_at_eos(new_tokens, accepted, EOS)
    check("EOS at final position: keeps the whole sequence", truncated == new_tokens)
    check("EOS at final position: accepted unchanged", kept_accepted == 3)


def test_budget_smaller_than_native_block():
    bs = compute_block_len(
        block_size=16,
        max_new_tokens=50,
        num_generated=47,
        max_seq_len=4096,
        position=100,
    )
    check("budget < native block: shrinks to remaining token budget", bs == 3)


def test_context_budget_smaller_than_native_block():
    bs = compute_block_len(
        block_size=16,
        max_new_tokens=1000,
        num_generated=0,
        max_seq_len=105,
        position=100,
    )
    check("context budget < native block: shrinks to remaining context", bs == 5)


def test_draft_cache_capacity_bounds_further():
    bs = compute_block_len(
        block_size=16,
        max_new_tokens=1000,
        num_generated=0,
        max_seq_len=4096,
        position=100,
        draft_room=4,
    )
    check("draft cache capacity is the binding constraint", bs == 4)


def test_generation_stops_when_block_len_hits_zero():
    bs = compute_block_len(
        block_size=16,
        max_new_tokens=50,
        num_generated=50,
        max_seq_len=4096,
        position=100,
    )
    check("exhausted token budget: block_len is 0 (caller should stop)", bs == 0)


if __name__ == "__main__":
    test_first_token_rejected()
    test_full_acceptance_with_bonus()
    test_partial_acceptance()
    test_eos_mid_block_truncates()
    test_eos_not_present_is_a_no_op()
    test_eos_at_final_position()
    test_budget_smaller_than_native_block()
    test_context_budget_smaller_than_native_block()
    test_draft_cache_capacity_bounds_further()
    test_generation_stops_when_block_len_hits_zero()
    print("\nALL PASS: acceptance loop synthetic tests.")
