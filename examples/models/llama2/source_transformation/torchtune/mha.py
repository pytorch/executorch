import torch
from torchtune.modules.attention.attention import MultiHeadAttention as TorchTuneMHA
from executorch.examples.models.llama2.source_transformation.torchtune.model import MultiHeadAttention as ExecuTorchMHA


def _replace_torchtune_mha_with_custom_mha(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, MultiHeadAttention):
            setattr(
                module,
                name,
                ExecuTorchMHA(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    num_kv_heads=child.num_kv_heads,
                    head_dim=child.head_dim,
                    q_proj=self.q_proj,
                    k_proj=self.k_proj,
                    v_proj=self.v_proj,
                    output_proj=self.output_proj,
                    pos_embeddings=self.pos_embeddings,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    kv_cache=self.kv_cache,
                    max_seq_len=self.max_seq_len,
                    is_causal=self.is_causal,
                    attn_dropout=self.attn_dropout,
                ),
            )
        else:
            _replace_torchtune_mha_with_custom_mha


def replace_torchtune_mha_with_custom_mha(module: torch.nn.Module) -> torch.nn.Module:
    _replace_torchtune_mha_with_custom_mha(module)
    return module
