from typing import Dict

from torchtune.training import FullModelHFCheckpointer
# from torchtune.models import convert_weights
from torchtune.models.convert_weights import get_mapped_key
import torch

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_QWEN_2_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wk.bias": "layers.{}.attn.k_proj.bias",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wq.bias": "layers.{}.attn.q_proj.bias",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wv.bias": "layers.{}.attn.v_proj.bias",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
    
}

def qwen_2_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _QWEN_2_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict

# TODO: no need to use TorchTune checkpointer, can just aggregate checkpoint files by ourselves.
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir='/home/jackzhxng/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/',
    checkpoint_files=['model.safetensors'],
    output_dir='.' ,
    model_type='QWEN2'
)

print("Loading checkpoint")
sd = checkpointer.load_checkpoint()

print("HF weights:")
for weight in sd["model"].keys():
    print(weight)
print()

# Convert from TorchTune to Meta (PyTorch native)
sd = qwen_2_tune_to_meta(sd['model'])

print("Meta weights:")
for weight in sd.keys():
    print(weight)

print("Saving checkpoint")
torch.save(sd, "/home/jackzhxng/models/qwen2_5-1_5b.pth") 
