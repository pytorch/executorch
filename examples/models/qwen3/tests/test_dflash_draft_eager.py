import torch
from executorch.backends.mlx.examples.llm.dflash_draft_model import DFlashConfig, DFlashDraftModel

config = DFlashConfig(
    hidden_size=2560,
    num_hidden_layers=5,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=9728,
    vocab_size=151936,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    max_position_embeddings=40960,
    target_layer_ids=(1, 9, 17, 25, 33),
    block_size=16,
    mask_token_id=151669,
    layer_types=("full_attention",) * 5,
)

model = DFlashDraftModel(config)
model.eval()

block_size, ctx_len = 16, 12
tokens = torch.randint(0, config.vocab_size, (1, block_size), dtype=torch.long)
target_hidden = torch.randn(1, ctx_len, len(config.target_layer_ids) * config.hidden_size)
position_ids = torch.arange(ctx_len + block_size).unsqueeze(0)

with torch.no_grad():
    logits = model(tokens, target_hidden, position_ids)

assert logits.shape == (1, block_size - 1, config.vocab_size), logits.shape
assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
print(f"OK: draft logits {tuple(logits.shape)}, no NaN/Inf")