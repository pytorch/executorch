# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline token embedding component.

Re-exports the token embedding module from the existing llama module, providing
a stable import path within the genai_pipeline namespace. Models that export
their embedding table as a separate graph construct it from here.

Usage:
    from executorch.backends.qualcomm.genai_pipeline.model_components.embedding import (
        TokenEmbedding,
    )
"""

from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import (
    TokenEmbedding,
)

__all__ = [
    "TokenEmbedding",
]
