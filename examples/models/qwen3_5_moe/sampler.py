"""
GPU-side Gumbel-max sampler.

Self-contained sampling utility that can be imported by other models. Lives
in its own file so it can be reused without pulling in the heavy MoE module.

``temperature`` is a runtime tensor so a single exported program can be
re-driven with different sampling configurations without re-export.

"""

from typing import Optional

import torch


def sample(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GPU-side Gumbel-max sampler.

    When ``temperature`` is ``None`` (the eager / eval default) the function
    is a no-op and returns ``logits`` unchanged — useful for callers that
    just want to inspect raw logits.

    Otherwise it draws from ``softmax(logits / temperature)`` entirely
    on-device using the Gumbel-max trick:
    ``argmax(logits / T + gumbel_noise)``
    (reference: https://huggingface.co/blog/cxdu/fastsampling).

    NOTE: the ``1e-20`` epsilons used in the Gumbel transform assume
    float32 logits. The contract is documented as ``[B, V]`` float32 and
    callers are expected to ``.float()``-cast before invoking ``sample``.

    TODO(gasoonjia): add top-k / top-p filtering support in a follow-up PR.

    Args:
        logits: ``[B, V]`` float32 logits.
        temperature: 0-D or 1-D float tensor (clamped to >= 1e-6 to avoid
            divide-by-zero). ``None`` skips temperature scaling and the
            sampler returns the unmodified ``logits`` tensor.

    Returns:
        ``[B, 1]`` float32 tensor of sampled token IDs, or the unmodified
        ``logits`` tensor when ``temperature`` is ``None``.
    """
    # No sampling configured — return raw logits.
    if temperature is None:
        return logits

    logits = logits / temperature.clamp(min=1e-6)

    # Gumbel-max sampling — equivalent to sampling from softmax(logits)
    # but fully on-device and CUDA-graph friendly. The 1e-20 epsilons are
    # safe for float32 (smallest positive normal ~1.18e-38) — see the
    # float32 note in the docstring.
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return (logits + gumbel).argmax(dim=-1, keepdim=True).float()
