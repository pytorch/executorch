# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verifies that the exported DFlash target model runs correctly and returns both logits and concatenated hidden states with the expected shapes. 

Run this after exporting the target model with --dflash-layers. 
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

tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
input_pos = torch.tensor([0], dtype=torch.long)
logits, hidden = method.execute([tokens, input_pos])

assert logits.shape == (1, 3, VOCAB_SIZE), logits.shape
assert hidden.shape == (1, 3, EXPECTED_HIDDEN_DIM), hidden.shape
assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
assert not torch.isnan(hidden).any() and not torch.isinf(hidden).any()

print(f"OK- logits {tuple(logits.shape)}, hidden {tuple(hidden.shape)}")
