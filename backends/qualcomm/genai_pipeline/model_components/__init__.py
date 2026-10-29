# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline model components.

Re-exports the exportable ``nn.Module`` building blocks from the existing llama
module, providing a stable import path within the genai_pipeline namespace:

.. code-block:: text

    model_components.decoder     <- llama.model.static_llama, .layernorm,
                                    .feed_forward, .apply_rope
    model_components.encoders    <- vision_encoder, audio_encoder
    model_components.embedding   <- token embedding

This package intentionally re-exports nothing itself. ``decoder`` pulls in
``static_llama`` and ``encoders`` pulls in the transformers vision/audio modeling
code, both of which are expensive to import; flattening them here would make
every consumer of any component pay for all of them. Import the submodule you
need instead:

Usage:
    from executorch.backends.qualcomm.genai_pipeline.model_components.decoder import (
        LlamaModel,
    )
    from executorch.backends.qualcomm.genai_pipeline.model_components.embedding import (
        TokenEmbedding,
    )
"""
