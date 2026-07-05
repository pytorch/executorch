"""Sanity check: qwen3_4b_dflash_target.pte returns (logits, hidden) with the
expected shapes at runtime.

Run after exporting with --dflash-layers, e.g.:
    python3 export_llm_hf.py --model-id Qwen/Qwen3-4B --dflash-layers 2,18,33 ...
    python3 test_dflash_export.py qwen3_4b_dflash_target.pte
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

print(f"OK: logits {tuple(logits.shape)}, hidden {tuple(hidden.shape)}")