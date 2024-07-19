import torch
import time

from transformers import Phi3Config, Phi3ForCausalLM, AutoTokenizer

torch.manual_seed(0)

end_of_text_token = 32000
max_length = 128

model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print("Model initialized")

tokens = tokenizer("tell me a story", return_tensors="pt").input_ids
print(f"Input ids: {tokens}")

start = time.time()
result = model.forward(input_ids=tokens, use_cache=True, return_dict=True)

current_token = torch.argmax(result.logits[:, -1, :], dim=-1).item()
current_key_value = result.past_key_values

print(f"Generating tokens: {current_token}", end='')

generated_tokens = [current_token]

while current_token != end_of_text_token and len(generated_tokens) < max_length:
    result = model.forward(
        input_ids=torch.tensor([[current_token]], dtype=torch.long),
        use_cache=True,
        return_dict=True,
        past_key_values=current_key_value)
    current_token = torch.argmax(result.logits[:, -1, :], dim=-1).item()
    current_key_value = result.past_key_values
    print(f", {current_token}", end='')
    generated_tokens.append(current_token)

print(f"\nGenerated response: \n{tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)}")

print(f"Took {time.time() - start}")


