"""
GPU-side Gumbel-max sampler with optional top-k / top-p filtering.

Self-contained sampling utility that can be imported by other models. Lives
in its own file so it can be reused without pulling in the heavy MoE module.

All sampling parameters (``temperature``, ``top_k``, ``top_p``) are
**runtime tensors** so a single exported program can be re-driven with
different sampling configurations without re-export.
"""

from typing import Optional

import torch


def sample(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    top_k: Optional[torch.Tensor] = None,
    top_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GPU-side Gumbel-max sampler with optional top-k / top-p filtering.

    All three sampling knobs are *runtime* scalar tensors so the caller can
    change them between calls without re-exporting the graph. The Python-
    level ``is None`` checks are static (decided at trace time) and select
    which subgraph is emitted; once provided, the actual values are pure
    tensors and the kernels are fully data-driven.

    When ``temperature``, ``top_k`` and ``top_p`` are all ``None`` (the
    eager / eval default), the function is a no-op and returns ``logits``
    unchanged — useful for callers that just want to inspect raw logits.

    Otherwise it draws from ``softmax(logits / temperature)`` entirely
    on-device using the Gumbel-max trick:
    ``argmax(logits / T + gumbel_noise)``
    (reference: https://huggingface.co/blog/cxdu/fastsampling).

    NOTE: the ``1e-20`` epsilons used in the Gumbel transform assume
    float32 logits. The contract is documented as ``[B, V]`` float32 and
    callers are expected to ``.float()``-cast before invoking ``sample``.

    Args:
        logits: ``[B, V]`` float32 logits.
        temperature: 0-D or 1-D float tensor (clamped to >= 1e-6 to avoid
            divide-by-zero). ``None`` skips temperature scaling.
        top_k: 0-D or 1-D int tensor — keep only the top ``k`` logits.
            ``None`` skips top-k filtering. ``k >= V`` is also a no-op.
        top_p: 0-D or 1-D float tensor — nucleus threshold; keep the
            smallest set of logits whose cumulative softmax probability
            is >= ``top_p``. ``None`` (or ``>= 1.0``) disables top-p.

    Returns:
        ``[B, 1]`` float32 tensor of sampled token IDs, or the unmodified
        ``logits`` tensor when all sampling parameters are ``None``.
    """
    # No sampling configured — return raw logits.
    if temperature is None and top_k is None and top_p is None:
        return logits

    if temperature is not None:
        logits = logits / temperature.clamp(min=1e-6)

    # Single sort handles both top-k and top-p filtering — both branches
    # need descending logits anyway, so we share the sort to keep the
    # graph small.
    if top_k is not None or top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_remove = torch.zeros_like(sorted_logits, dtype=torch.bool)

        if top_k is not None:
            # Position >= k → drop. Works for any tensor k via broadcast;
            # k >= V naturally becomes a no-op (mask is all-False).
            pos = torch.arange(sorted_logits.size(-1), device=sorted_logits.device)
            sorted_remove = sorted_remove | (pos >= top_k.to(pos.dtype))

        if top_p is not None:
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            p_remove = cum_probs > top_p
            # Shift right by one so the highest-prob token is always kept,
            # even when its single-token prob already exceeds top_p.
            p_remove = torch.cat(
                [torch.zeros_like(p_remove[..., :1]), p_remove[..., :-1]],
                dim=-1,
            )
            sorted_remove = sorted_remove | p_remove

        sorted_logits = torch.where(
            sorted_remove,
            torch.full_like(sorted_logits, float("-inf")),
            sorted_logits,
        )
        # Scatter the masked sorted logits back into original token order.
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    # Gumbel-max sampling — equivalent to sampling from softmax(logits)
    # but fully on-device and CUDA-graph friendly. The 1e-20 epsilons are
    # safe for float32 (smallest positive normal ~1.18e-38) — see the
    # float32 note in the docstring.
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return (logits + gumbel).argmax(dim=-1, keepdim=True).float()
