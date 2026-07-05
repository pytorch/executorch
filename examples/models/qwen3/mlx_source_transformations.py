"""Extracting Qwen3 hidden-state for DFlash.

Same idea as examples/models/gemma4_31b/mlx_source_transformations.py --
tap layers [2, N//2, N-3] and return them concatenated alongside logits.
Gemma 4 does this by patching its own hand-written forward(). Qwen3 goes
through the generic HF export path instead (export_llm_hf.py), which wraps
the model in transformers' TorchExportableModuleWithStaticCache before
torch.export. So we subclass that wrapper and add output_hidden_states
to its forward rather than patching Qwen3 itself.

Base class signature/behavior confirmed via:
    inspect.getsource(transformers.integrations.executorch.TorchExportableModuleWithStaticCache)
"""

from typing import List, Optional, Sequence

import torch
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache


class TorchExportableModuleWithStaticCacheAndHidden(TorchExportableModuleWithStaticCache):
    """forward() also returns tapped hidden states.
    forward() -> (logits, hidden), where hidden is [B, T, len(layer_ids) * H].
    """

    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_ids: Sequence[int] = (),
    ):
        super().__init__(model, batch_size=batch_size, max_cache_len=max_cache_len, device=device)
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.static_cache,
            use_cache=True,
            output_hidden_states=True,
        )

        # hidden_states[0] is the embedding output, hidden_states[i+1] is decoder layer i's output
        captured = [outs.hidden_states[i + 1] for i in self.layer_ids]
        hidden = torch.cat(captured, dim=-1)

        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden


def default_dflash_layer_ids(num_layers: int) -> List[int]:
    """[2, N//2, N-3] tap pattern, same as Gemma 4. For Qwen3-4B (36 layers): [2, 18, 33]."""
    return [2, num_layers // 2, num_layers - 3]

class StatelessQwen3WithHidden(torch.nn.Module):
    """Cache-free counterpart to TorchExportableModuleWithStaticCacheAndHidden.

    DFlash's speculative-decode loop re-verifies overlapping/non-contiguous
    token ranges every round (draft tokens get proposed, verified, some
    rejected). TorchExportableModuleWithStaticCache's persistent internal
    cache is built for strictly-sequential autoregressive decoding and
    produces corrupted hidden states under this access pattern (confirmed at
    the eager PyTorch level: two calls at non-contiguous cache_position values
    on the same cached wrapper differ by ~8694 vs. ~0.0003 for a correctly
    stateless model). This class sidesteps the whole problem: every forward()
    call recomputes attention over exactly the tokens/positions given, with no
    persistent state, matching the accumulate-full-context approach already
    used by DFlashDraftModel in dflash_draft_model.py.

    forward() -> (logits, hidden), where hidden is [B, T, len(layer_ids) * H].
    """

    def __init__(self, model, layer_ids: Sequence[int] = ()):
        super().__init__()
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.model = model
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        outs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
        )
        captured = [outs.hidden_states[i + 1] for i in self.layer_ids]
        hidden = torch.cat(captured, dim=-1)
        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden
