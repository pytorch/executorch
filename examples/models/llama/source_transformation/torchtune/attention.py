import torch
import torchtune.modules.attention as TorchTuneAttention
from executorch.examples.models.llama2.source_transformation.torchtune.modules.mha import (
    MultiHeadAttention,
)


def _replace_mha_with_inference_mha(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, TorchTuneAttention.MultiHeadAttention):
            setattr(
                module,
                name,
                MultiHeadAttention(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    num_kv_heads=child.num_kv_heads,
                    head_dim=child.head_dim,
                    q_proj=child.q_proj,
                    k_proj=child.k_proj,
                    v_proj=child.v_proj,
                    output_proj=child.output_proj,
                    pos_embeddings=child.pos_embeddings,
                    q_norm=child.q_norm,
                    k_norm=child.k_norm,
                    kv_cache=child.kv_cache,
                    max_seq_len=child.max_seq_len,
                    is_causal=child.is_causal,
                    attn_dropout=child.attn_dropout,
                ),
            )
        else:
            replace_mha_with_inference_mha(child)


def replace_mha_with_inference_mha(module: torch.nn.Module) -> torch.nn.Module:
    """
    Replace TorchTune's MHA with an inference friendly version of MHA that
    separates out the inference-related parts for further optimization.
    """
    _replace_mha_with_inference_mha(module)
    return module
