import unittest
from typing import Optional

import torch
from executorch.examples.models.llama2._static_llama2.static_llama import (
    LlamaForCausalLM,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


class StaticLlama2ExportTest(unittest.TestCase):
    def test_static_llama(self):
        torch.manual_seed(0)
        checkpoint_path = ""
        dim = 64
        multiple_of = 4
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        intermediate_size = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )
        # checkpoint_path = "/data/sandcastle/boxes/fbsource/fbcode/executorch/examples/models/llama2/params/demo_rand_params.pth"
        # params_path = "/data/sandcastle/boxes/fbsource/fbcode/executorch/examples/models/llama2/params/demo_config.json"
        checkpoint_path = "/home/chenlai/local/very_new_checkpoint.pt"
        params_path = "/home/chenlai/local/lang_params.json"
        llama_for_causal_lm = LlamaForCausalLM(
            checkpoint=checkpoint_path, params=params_path
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        # pop_list = [
        #     "layers.0.attention.mask",
        #     "layers.1.attention.mask",
        #     "layers.2.attention.mask",
        #     "layers.3.attention.mask",
        #     "layers.4.attention.mask",
        # ]
        # for redundant_weight in pop_list:
        #     checkpoint.pop(redundant_weight)

        # llama_for_causal_lm.model.load_state_dict(checkpoint)
        prompt_tokens = torch.tensor([[3, 1, 2, 4, 3]])
        all_inputs = llama_for_causal_lm.get_all_inputs(prompt_tokens)

        out = llama_for_causal_lm(**all_inputs)
        next_token = sample(out[0])[0][0].item()

        print(next_token)
        self.assertEqual(0, 1)
