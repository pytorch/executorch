# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registry mapping a checkpoint's ``model_type`` to its DFlash adapter.

Adapters hold the one part of the pipeline that genuinely differs per
checkpoint: draft weight loading, which depends on the keys the draft omits and
where they are copied from in the target. Everything else in the DFlash export
and runner path is architecture-agnostic -- chat templating in particular is
carried by the checkpoint's own tokenizer config, so it lives in runtime_meta.
"""

from typing import Callable, Dict, NamedTuple


class DFlashAdapter(NamedTuple):
    load_draft_weights: Callable


_ADAPTERS: Dict[str, DFlashAdapter] = {}


def register_adapter(model_type: str, *, load_draft_weights: Callable) -> None:
    _ADAPTERS[model_type] = DFlashAdapter(load_draft_weights=load_draft_weights)


def get_adapter(model_type: str) -> DFlashAdapter:
    try:
        return _ADAPTERS[model_type]
    except KeyError:
        raise ValueError(
            f"No DFlash adapter registered for model_type {model_type!r}. "
            f"Registered: {sorted(_ADAPTERS)}."
        ) from None


def load_qwen3_dflash_draft_weights(draft_model, checkpoint_path, target_state_dict):
    """Load Qwen3 DFlash checkpoint weights and copy shared target weights.
    The draft checkpoint omits its embedding and LM head, which are copied from the target.
    """
    from safetensors.torch import load_file

    draft_weights = {}
    for f in checkpoint_path.glob("*.safetensors"):
        draft_weights.update(load_file(str(f)))

    missing, unexpected = draft_model.load_state_dict(draft_weights, strict=False)
    assert not unexpected, f"Unexpected draft checkpoint keys: {unexpected}"
    still_missing = [
        k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))
    ]
    assert not still_missing, f"Missing draft checkpoint keys: {still_missing}"

    draft_model.embed_tokens.weight.data.copy_(
        target_state_dict["model.embed_tokens.weight"]
    )
    lm_head_key = (
        "lm_head.weight"
        if "lm_head.weight" in target_state_dict
        else "model.embed_tokens.weight"
    )
    draft_model.lm_head.weight.data.copy_(target_state_dict[lm_head_key])


register_adapter(
    "qwen3",
    load_draft_weights=load_qwen3_dflash_draft_weights,
)
