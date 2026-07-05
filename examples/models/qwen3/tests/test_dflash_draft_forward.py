from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from executorch.backends.mlx.examples.llm.dflash_draft_model import DFlashDraftModel, load_dflash_config

path = Path(snapshot_download("z-lab/Qwen3-4B-DFlash-b16", allow_patterns=["*.safetensors", "*.json"]))
config = load_dflash_config(path)

model = DFlashDraftModel(config)
weights = {}
for f in path.glob("*.safetensors"):
    weights.update(load_file(str(f)))
model.load_state_dict(weights, strict=False)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype="auto")
model.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight)
model.lm_head.weight.data.copy_(target.lm_head.weight)
model.eval()
target.eval()

prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    out = target(input_ids, output_hidden_states=True)
    tapped = [out.hidden_states[i + 1] for i in config.target_layer_ids]
    target_hidden = torch.cat(tapped, dim=-1).float()

    last_token = input_ids[:, -1:]
    block_size = 8
    draft_tokens = torch.cat(
        [last_token, torch.full((1, block_size - 1), config.mask_token_id, dtype=torch.long)], dim=1
    )
    position_ids = torch.arange(target_hidden.shape[1] + block_size).unsqueeze(0)

    draft_logits = model(draft_tokens, target_hidden, position_ids)
    predicted = draft_logits.argmax(-1)

print("Prompt:", prompt)
print("Predicted continuation:", tokenizer.decode(predicted[0]))
print("Shape:", draft_logits.shape)
