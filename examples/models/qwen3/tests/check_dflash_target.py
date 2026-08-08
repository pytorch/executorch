# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verifies that the exported DFlash target model runs correctly and returns
both logits and concatenated hidden states with the expected shapes.

Run this after exporting the target model with --tap-layers.
"""

import sys

import torch
from executorch.runtime import Runtime, Verification

DFLASH_LAYERS = [1, 9, 17, 25, 33]
HIDDEN_SIZE = 2560
EXPECTED_HIDDEN_DIM = len(DFLASH_LAYERS) * HIDDEN_SIZE  # 12800
VOCAB_SIZE = 151936

pte_path = sys.argv[1]
et_runtime = Runtime.get()
program = et_runtime.load_program(pte_path, verification=Verification.Minimal)
method = program.load_method("forward")

# Exercise multiple sequence lengths, including seq_len=1 -- tokens and
# cache positions share one exported dynamic symbol, so a single fixed
# length wouldn't catch a mismatch between them.
for seq_len in [1, 3, 8, 16]:
    tokens = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    # One cache position PER TOKEN, not a single fixed position -- tokens
    # and input_pos must have matching length, since they share one
    # exported dynamic dimension. The original version of this check
    # passed 3 tokens with a single-element input_pos, which happened not
    # to crash but silently exercised an inconsistent shape pairing.
    input_pos = torch.arange(seq_len, dtype=torch.long)
    logits, hidden = method.execute([tokens, input_pos])

    assert logits.shape == (1, seq_len, VOCAB_SIZE), (seq_len, logits.shape)
    assert hidden.shape == (1, seq_len, EXPECTED_HIDDEN_DIM), (seq_len, hidden.shape)
    assert not torch.isnan(logits).any() and not torch.isinf(logits).any(), seq_len
    assert not torch.isnan(hidden).any() and not torch.isinf(hidden).any(), seq_len

    print(f"OK seq_len={seq_len}: logits {tuple(logits.shape)}, hidden {tuple(hidden.shape)}")

print("PASS: target smoke check passed for all tested sequence lengths.")
