# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict transforms that rewrite the checkpoint's key layout.

None of these can mutate in place: they return a different dict -- a rebuilt one
for the renames, the nested one for the unwrap -- while the transforms elsewhere
in this package rewrite values.
"""

from __future__ import annotations

from typing import Any, Dict


def unwrap_model_key(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Take the weights out of a checkpoint that nests them under ``model``.

    Local checkpoints wrap the weights alongside their training state;
    HF-converted ones are already flat. Self-guarding on the key, so no model has
    to opt out and this can lead every model's chain -- it has to, since every
    rename after it looks keys up by their canonical name.
    """
    return state_dict["model"] if "model" in state_dict else state_dict


def strip_orig_mod_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the ``_orig_mod.`` prefix ``torch.compile`` leaves on checkpoints."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def remap_gemma4_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Rename Gemma4's converted weights to the static decoder's naming."""
    from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.convert_weights import (
        remap_keys,
    )

    return remap_keys(state_dict)
