# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time

from transformers import Phi3ForCausalLM, AutoTokenizer

torch.manual_seed(0)

end_of_text_token = 32000
max_length = 128

model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

input_ids = tokenizer("tell me a story", return_tensors="pt").input_ids

print("Input ids: {}".format(input_ids))

next_token = 0
generated_tokens = []

start = time.time()
for _ in range(max_length):
    outputs = model.forward(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    generated_tokens.append(next_token.item())

    if next_token == end_of_text_token:
        break

    input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

print("Generated ids: \n {}".format(generated_tokens))
print("Generated response: \n {}".format(
    tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
))
print(f"Took {time.time() - start}")