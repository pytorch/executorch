import torch
import torchtune.modules.attention as TorchTuneAttention
from executorch.examples.models.llama2.source_transformation.torchtune.modules.mha import MultiHeadAttention
from executorch.examples.models.llama2.source_transformation.torchtune.modules.sdpa import SDPA

def _replace_mha_with_inference_mha(module: torch.nn.Module):
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
                    pos_embeddings=child.pos_embedding,
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

def replace_mha_with_inference_mha(module: torch.nn.Module):
    """
    Replace TorchTune's MHA with an inference friendly version of MHA that
    separates out the inference-related parts for further optimization.
    """
    _replace_mha_with_inference_mha(module)
    return module

# class SDPACustom(torch.nn.Module):
#     def __init__(
#         self,
#         kv_cache: KVCache,
#         dim: int,
#     ):
#         super().__init__()
#         # Custom op only supports float32 currently. Converting to/from float32 is
#         # faster than not having the op.
#         self.kv_cache = kv_cache.to(torch.float)
#         self.dim = dim

#     def forward(
#         self,
#         input_pos: torch.Tensor,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         bsz,
#         seqlen,
#         mask,
#     ):
#         # Custom op only supports float32 currently. Converting to/from float32 is
#         # faster than not having the op.
#         input_dtype = q.dtype
#         q = q.to(dtype=torch.float)
#         k = k.to(dtype=torch.float)
#         v = v.to(dtype=torch.float)
#         output = torch.ops.llama.sdpa_with_kv_cache(
#             q,
#             k,
#             v,
#             self.kv_cache.k_cache,
#             self.kv_cache.v_cache,
#             input_pos[-1].item(),
#             seqlen,
#             None,  # Attention mask
#             0,  # dropout probability. Ignored by the code
#             True,  # is_causal
#         )
#         return output.view(bsz, seqlen, self.dim).to(dtype=input_dtype)


# def _replace_sdpa_with_custom_op(module: torch.nn.Module):
#     for name, child in module.named_children():
#         if isinstance(child, SDPA):
#             setattr(
#                 module,
#                 name,
#                 SDPACustom(child.kv_cache, child.dim),
#             )
#         else:
#             _replace_sdpa_with_custom_op(child)


# def replace_sdpa_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:
#     from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa

#     _replace_sdpa_with_custom_op(module)
#     return module

