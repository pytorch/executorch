from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, StaticCache
import torch

model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id).eval()
config = Qwen2Config.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Hello, how are you?"
max_seq_len = 128
past_key_values = StaticCache(config=config, max_batch_size=1,  max_cache_len=max_seq_len)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_tokens = input_ids[0].tolist()
position_ids = torch.tensor([0]).unsqueeze(0)
attention_mask = torch.full((1, 1, 1, max_seq_len), torch.finfo(torch.float32).min)

# huggingface version
with torch.no_grad():
  tokens = 1
  while tokens < max_seq_len:
    cur_pos = torch.tensor([tokens - 1], dtype=torch.long)
    attention_mask[:, :, :, cur_pos] = 0
    outputs = model(
      input_ids=torch.tensor([generated_tokens[cur_pos]]).unsqueeze(0),
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      num_logits_to_keep=1,
    )

    if tokens >= len(generated_tokens):
      generated_tokens.append(outputs.logits.argmax(dim=-1).item())
    if generated_tokens[-1] == tokenizer.eos_token_id:
      break
    tokens += 1
    position_ids[0, 0] += 1

print(f"hf_model output: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}")

# qc version
from executorch.examples.qualcomm.oss_scripts.qwen.model.static_qwen import Qwen2ForCausalLM
from executorch.examples.qualcomm.oss_scripts.llama.llama import smart_mask_updater
import tempfile
qc_model = Qwen2ForCausalLM(config)
with tempfile.TemporaryDirectory() as tmp_dir:
  pt_file = f"{tmp_dir}/hf_weights.pt"
  torch.save(model.state_dict(), pt_file)
  qc_model.load_state_dict(torch.load(pt_file, weights_only=True))
  qc_model.model.norm.prepare_torch_rms_norm()
  for layer in qc_model.model.layers:
    layer.self_attn.prepare_sha()
    layer.input_layernorm.prepare_torch_rms_norm()
    layer.post_attention_layernorm.prepare_torch_rms_norm()
    layer.mlp.prepare_feedfoward_conv()

_, atten_mask, _, k_caches, v_caches = qc_model.get_example_inputs()
all_pos = torch.arange(0, max_seq_len, 1, dtype=torch.int32).unsqueeze(0)
token_list = input_ids[0].tolist()
pos, ar_len = 1, 1

with torch.no_grad():
  while token_list[-1] != tokenizer.eos_token_id and pos < max_seq_len:
    tmp_token_list = torch.tensor(token_list[pos - ar_len : pos], dtype=torch.int32).reshape(1, -1)
    tmp_pos = all_pos[:, pos - ar_len : pos]
    tmp_atten_mask = atten_mask
    logits, new_k_caches, new_v_caches = qc_model(
      tmp_token_list,
      tmp_atten_mask,
      tmp_pos,
      *k_caches,
      *v_caches,
    )
    atten_mask, pos, k_caches, v_caches = smart_mask_updater(
      ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
    )
    if pos > len(token_list):
      token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

print(f"qc_model output: {tokenizer.decode(token_list)}")
