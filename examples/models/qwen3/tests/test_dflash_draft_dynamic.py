import torch
from executorch.runtime import Runtime, Verification

rt = Runtime.get()
method = rt.load_program("qwen3_4b_dflash_draft.pte",
                         verification=Verification.Minimal).load_method("forward")

block_size, hidden_size, vocab_size = 16, 12800, 151936
for ctx_len in (8, 20, 1):
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    target_hidden = torch.randn(1, ctx_len, hidden_size)
    position_ids = torch.arange(ctx_len + block_size).unsqueeze(0).long()
    (draft_logits,) = method.execute([tokens, target_hidden, position_ids])
    assert draft_logits.shape == (1, block_size - 1, vocab_size), (ctx_len, draft_logits.shape)
    assert not torch.isnan(draft_logits).any() and not torch.isinf(draft_logits).any()
    print(f"ctx_len={ctx_len}: OK {tuple(draft_logits.shape)}")

print("PASS: dynamic ctx_len verified at multiple lengths")
