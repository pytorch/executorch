# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-specific adapters for the generic DFlash export and runner scripts. 

These handle Qwen3-specific checkpoint weight keys and chat-template behavior, keeping model-specific logic separate from the shared export and inference pipeline. 
"""


def load_qwen3_dflash_draft_weights(draft_model, checkpoint_path, target_state_dict):
    """Load Qwen3 DFlash checkpoint weights and copy shared target weights. 
    The draft checkpoint omits its embedding and LM head, which are copied from the target."""
    from safetensors.torch import load_file

    draft_weights = {}
    for f in checkpoint_path.glob("*.safetensors"):
        draft_weights.update(load_file(str(f)))

    missing, unexpected = draft_model.load_state_dict(draft_weights, strict=False)
    assert not unexpected, f"Unexpected draft checkpoint keys: {unexpected}"
    still_missing = [k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))]
    assert not still_missing, f"Missing draft checkpoint keys: {still_missing}"

    draft_model.embed_tokens.weight.data.copy_(target_state_dict["model.embed_tokens.weight"])
    lm_head_key = (
        "lm_head.weight" if "lm_head.weight" in target_state_dict else "model.embed_tokens.weight"
    )
    draft_model.lm_head.weight.data.copy_(target_state_dict[lm_head_key])


def apply_qwen3_chat_template(tokenizer, prompt, enable_thinking):
    """Apply Qwen3's chat template with its model-specific thinking option."""
    messages = [{"role": "user", "content": prompt}]
    chat_out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    # Different Transformers versions return either a BatchEncoding or a tensor.
    return chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out
