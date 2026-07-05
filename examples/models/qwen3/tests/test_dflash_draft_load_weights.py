import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from executorch.backends.mlx.examples.llm.dflash_draft_model import DFlashConfig, DFlashDraftModel

path = Path(snapshot_download("z-lab/Qwen3-4B-DFlash-b16", allow_patterns=["*.safetensors", "*.json"]))
cfg = json.loads((path / "config.json").read_text())
dcfg = cfg["dflash_config"]

config = DFlashConfig(
    hidden_size=cfg["hidden_size"],
    num_hidden_layers=cfg["num_hidden_layers"],
    num_attention_heads=cfg["num_attention_heads"],
    num_key_value_heads=cfg["num_key_value_heads"],
    head_dim=cfg["head_dim"],
    intermediate_size=cfg["intermediate_size"],
    vocab_size=cfg["vocab_size"],
    rms_norm_eps=cfg["rms_norm_eps"],
    rope_theta=cfg["rope_theta"],
    max_position_embeddings=cfg["max_position_embeddings"],
    target_layer_ids=tuple(dcfg["target_layer_ids"]),
    block_size=cfg["block_size"],
    mask_token_id=dcfg["mask_token_id"],
    layer_types=tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]),
    sliding_window=cfg.get("sliding_window"),
    final_logit_softcapping=cfg.get("final_logit_softcapping"),
)

model = DFlashDraftModel(config)

draft_weights = {}
for f in path.glob("*.safetensors"):
    draft_weights.update(load_file(str(f)))

print(f"Checkpoint has {len(draft_weights)} tensors")
print("First 10 checkpoint keys:", list(draft_weights.keys())[:10])
print("First 10 model keys:     ", list(model.state_dict().keys())[:10])

missing, unexpected = model.load_state_dict(draft_weights, strict=False)
still_missing = [k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))]

print(f"\nMissing (excl. embed/lm_head, expected empty): {still_missing}")
print(f"Unexpected (expected empty): {unexpected}")

assert not still_missing, "Architecture mismatch — key names don't match the real checkpoint"
assert not unexpected, "Checkpoint has tensors our model doesn't define — architecture mismatch"
print("\nOK: state_dict loaded cleanly, structure matches the real checkpoint")