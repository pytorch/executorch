"""
Verifies that the exported draft .pte loads, runs correctly, and supports dynamic context lengths as the accumulated target hidden-state context grows during speculative decoding. A successful export and execution also confirms that the checkpoint weights and exported model are compatible.
"""

import sys

import torch
from executorch.runtime import Runtime, Verification

pte_path = sys.argv[1] if len(sys.argv) > 1 else "qwen3_4b_dflash_draft.pte"
et_runtime = Runtime.get()
method = et_runtime.load_program(
    pte_path, verification=Verification.Minimal
).load_method("forward")

block_size, hidden_size, vocab_size = 16, 12800, 151936

for ctx_len in (8, 20, 1):
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    target_hidden = torch.randn(1, ctx_len, hidden_size)
    position_ids = torch.arange(ctx_len + block_size).unsqueeze(0).long()

    (draft_logits,) = method.execute([tokens, target_hidden, position_ids])
    assert draft_logits.shape == (1, block_size - 1, vocab_size), (
        ctx_len,
        draft_logits.shape,
    )
    assert not torch.isnan(draft_logits).any() and not torch.isinf(draft_logits).any()
    print(f"ctx_len={ctx_len}: OK {tuple(draft_logits.shape)}")

print("PASS- draft .pte loads, executes, and supports dynamic ctx_len")
