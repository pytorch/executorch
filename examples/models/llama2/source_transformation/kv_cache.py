import torch
from executorch.examples.models.llama2.llama_transformer import KVCache
from typing import Tuple

class DynamicShapeKVCache(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        transpose_cache: bool,
        dtype=torch.float32,
    ):
        super().__init__()
        if transpose_cache:
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        else:
            cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)

        self.transpose_cache = transpose_cache
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.max_seq_length = max_seq_length

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, H, S, D] or [B, S, H, D] depending on transpose_cache
        start_pos = input_pos[-1].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos < self.max_seq_length)
        seq_length = k_val.size(2)
        # Replace the entry in the cache for this token
        # The following lines are equivalent to:
        # cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        # We use .narrow() here to make the compiler happy
        # pyre-ignore: Incompatible parameter type [6]
        narrowed_k = self.k_cache.narrow(2, start_pos, seq_length)
        # pyre-ignore: Incompatible parameter type [6]
        narrowed_v = self.v_cache.narrow(2, start_pos, seq_length)

        narrowed_k.copy_(k_val)
        narrowed_v.copy_(v_val)
        return self.k_cache, self.v_cache

def _replace_kv_cache_with_dynamic_kv_cache(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            print("find KVCche...")
            print(child)
            setattr(
                module,
                name,
                DynamicShapeKVCache(
                    max_batch_size=child.max_batch_size,
                    max_seq_length=child.max_seq_length,
                    n_heads=child.n_heads,
                    head_dim=child.head_dim,
                    transpose_cache=child.transpose_cache, 
                    dtype=child.dtype
                ),
            )
        else:
            _replace_kv_cache_with_dynamic_kv_cache(child)


def replace_kv_cache_with_dynamic_kv_cache(module: torch.nn.Module) -> torch.nn.Module:

    _replace_kv_cache_with_dynamic_kv_cache(module)
    return module
