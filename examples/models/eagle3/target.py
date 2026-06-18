# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Target-model abstraction for EAGLE-3 speculative decoding.

The EAGLE-3 machinery (draft head, speculator, export, runner) is target
agnostic. A target only has to (1) be an ExecuTorch-exportable model that exposes
hidden-state taps via the ``TapTarget`` protocol and (2) register a
``TargetSpec`` (how to load it + its export-shape constraints) in ``TARGETS``.

To add a target ``foo``:
  - implement ``set_eagle_tap_layers`` + ``forward_logits_taps`` on foo's model
    (collect the EAGLE-3 aux hidden states; HF/vLLM convention: index 0 =
    embedding output, index k = output after decoder layer k-1), and
  - add a ``TargetSpec`` entry to ``TARGETS`` with a loader and shape hints.
The draft head, scheme, kernels, and runner loop are unchanged.

The target, the EAGLE-3 draft head, and the tokenizer must be a matched set that
were trained together: the draft must be trained on this target's hidden states,
share its tokenizer/vocab and the d2t/t2d mapping, and use the same tap-layer
convention. Only target/draft hidden size is checked at export; the rest is the
caller's responsibility (a mismatch can run yet silently degrade acceptance).

gemma4-31B is the reference implementation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class TapTarget(Protocol):
    """A target LM instrumented with EAGLE-3 hidden-state taps.

    ``config`` must expose ``max_seq_len``, ``vocab_size`` and
    ``num_hidden_layers``. The two methods mirror the gemma4-31B reference:
    """

    config: Any

    def set_eagle_tap_layers(self, layers: list) -> None:
        """Select the aux hidden-state layers to collect (ascending indices)."""
        ...

    def forward_logits_taps(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        last_logits_only: bool = True,
    ):
        """Return (logits, taps): logits (B,1|T,vocab); taps (B,T,len(aux)*hidden)."""
        ...


@dataclass(frozen=True)
class TargetSpec:
    """How to load a target for export and its export-shape constraints."""

    # (target_dir, max_seq_len, backend) -> a CPU TapTarget with runtime buffers
    # materialized (export keeps the model on the host). ``backend`` selects the
    # weight packing ("cuda" or "mlx").
    load: Callable[..., TapTarget]
    # config -> max tokens accepted in one target forward (e.g. a sliding ring
    # buffer caps it at 2*window; a flat-cache model uses max_seq_len-1).
    max_forward_len: Callable[[Any], int]
    # Minimum tokens in ANY single target forward the export accepts (some
    # attention-mask implementations specialize a lower bound under
    # torch.export). Applies to both prefill and the static target_verify window.
    min_forward_len: int


def _load_gemma4_31b(
    target_dir: str, max_seq_len: int, backend: str = "cuda"
) -> TapTarget:
    from executorch.examples.models.gemma4_31b.export import load_prequantized_model
    from executorch.examples.models.gemma4_31b.model import materialize_runtime_buffers

    target, _ = load_prequantized_model(
        target_dir, max_seq_len=max_seq_len, backend=backend
    )
    materialize_runtime_buffers(target, dtype=torch.bfloat16, device="cpu")
    return target.eval()


TARGETS: dict[str, TargetSpec] = {
    "gemma4_31b": TargetSpec(
        load=_load_gemma4_31b,
        # Sliding ring buffer caps a single forward at 2*window.
        max_forward_len=lambda cfg: cfg.sliding_window * 2,
        # The gemma4 sliding-window mask specializes seq_len >= 5 under export.
        min_forward_len=5,
    ),
}
