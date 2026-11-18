# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model preparation transforms, declared per model by ``SOURCE_TRANSFORMS``.

Model preparation is *declared*, not subclassed. The reference flow embeds it in
``LLMWrapper._prepare_model()``, where overlapping transform sets (Llama needing
``[A, B, C]`` while Gemma needs ``[B, C, D]``) cannot be referenced
individually. Here each transform is a named function implemented once, and
enabling a model is a row in ``models.model_registry`` rather than a new adapter
class.

Ordering between the two transform stages is load-bearing, so a single
``transforms`` list is insufficient. The stages run:

.. code-block:: text

    construct module          (model_class_name)
          |
    load / convert checkpoint
          |
    state_dict_transforms
          |                     (with model-specific values already bound)
    load_state_dict(assign=True)
          |
    module_transforms

Each transform receives only its operand at execution time. Model-specific
construction in :func:`model_lookup.get_source_transform` binds every additional
dependency as an explicit value: RoPE receives layer and head counts, embedding
scaling receives its scale factor, and dtype conversion receives the requested
dtype. A transform therefore never needs the full model configuration or CLI
namespace.

* ``state_dict_transforms``: ``(state_dict) -> state_dict``. Transforms that only
  rewrite values mutate and return the same dict -- these state dicts hold
  multi-GB of weights and copying them is not free. Transforms that rename keys
  necessarily build a new dict.
* ``module_transforms``: ``(module) -> module``. In-place mutations return the
  same module; module-to-module replacements return the new one, so callers must
  always use the return value.
"""

from executorch.backends.qualcomm.genai_pipeline.source_transform.checkpoint_key_remap import (
    remap_gemma4_keys,
    strip_orig_mod_prefix,
    unwrap_model_key,
)
from executorch.backends.qualcomm.genai_pipeline.source_transform.dtype_override import (
    apply_dtype_override,
)
from executorch.backends.qualcomm.genai_pipeline.source_transform.embedding_scale import (
    scale_token_embedding,
)
from executorch.backends.qualcomm.genai_pipeline.source_transform.linear_to_conv2d import (
    convert_linear_to_conv2d,
    prepare_conv_submodules,
)
from executorch.backends.qualcomm.genai_pipeline.source_transform.rms_norm_offset import (
    gemma_rmsnorm_offset,
)
from executorch.backends.qualcomm.genai_pipeline.source_transform.rope_layout import (
    permute_partial_rope,
)

__all__ = [
    # state_dict transforms
    "gemma_rmsnorm_offset",
    "permute_partial_rope",
    "remap_gemma4_keys",
    "scale_token_embedding",
    "strip_orig_mod_prefix",
    "unwrap_model_key",
    # module transforms
    "apply_dtype_override",
    "convert_linear_to_conv2d",
    "prepare_conv_submodules",
]
