import sys
import torch
from executorch.runtime import Runtime, Verification

pte_path = sys.argv[1] if len(sys.argv) > 1 else "qwen3_4b_dflash_draft.pte"
et_runtime = Runtime.get()
program = et_runtime.load_program(pte_path, verification=Verification.Minimal)
method = program.load_method("forward")

# Must match the exact static shapes used at export time: ctx_len=8, block_size=16
block_size, ctx_len, hidden_size, vocab_size = 16, 8, 12800, 151936
tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
target_hidden = torch.randn(1, ctx_len, hidden_size)
position_ids = torch.arange(ctx_len + block_size).unsqueeze(0).long()

(draft_logits,) = method.execute([tokens, target_hidden, position_ids])
assert draft_logits.shape == (1, block_size - 1, vocab_size), draft_logits.shape
assert not torch.isnan(draft_logits).any() and not torch.isinf(draft_logits).any()
print(f"OK: draft_logits {tuple(draft_logits.shape)}")
