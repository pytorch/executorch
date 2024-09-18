import torch
from torchtune.modules.attention.sdpa import SDPA


class SDPACustom(nn.Module):
    def __init__(
        self,
        attn_dropout: float,
        is_causal: bool,
        kv_cache,
    ) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout
        self.is_causal = is_causal
        self._kv_cache = kv_cache

    def kv_cache_update(
        self,
        input_pos: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # TODO: unimplemented.
        return k, v

    def sdpa(
        self,
        q: Tensor,  # [b, s, n_h, h_d]
        k: Tensor,  # [b, s, n_kv, h_d]
        v: Tensor,  # [b, s, n_kv, h_d]
        input_pos: Tensor,
        bsz: int,
        seq_len: int,
        mask: Tensor = None,
    ) -> Tensor:
        output = torch.ops.llama.sdpa_with_kv_cache(
            q,
            k,
            v,
            self._kv_cache.k_cache,
            self._kv_cache.v_cache,
            input_pos[-1].item(),
            seq_len,
            mask,  # Attention mask
            self.attn_dropout,  # dropout probability. Ignored by the code
            self.is_causal,  # is_causal
        )
        return output.view(bsz, seqlen, -1)


def _replace_sdpa_with_custom_op(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, SDPA):
            setattr(
                module,
                name,
                SDPACustom(child.attn_dropout, child.is_causal, child.kv_cache),
            )
        else:
            _replace_sdpa_with_custom_op(child)


def replace_sdpa_with_custom_op(module: torch.nn.Module) -> torch.nn.Module:
    from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa

    _replace_sdpa_with_custom_op(module)
    return module
