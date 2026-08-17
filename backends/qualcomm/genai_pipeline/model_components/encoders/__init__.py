# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline multimodal encoder components.

Re-exports the vision and audio encoders from the existing llama module,
providing a stable import path within the genai_pipeline namespace. Each class
is the encoder graph of one modality, paired with a text decoder by
``llama.encoder.encoder_config``.

The ``Custom*`` classes those encoders swap in via
``replace_module_with_custom_class`` are export workarounds internal to their own
modules, and are deliberately not re-exported here.

Usage:
    from executorch.backends.qualcomm.genai_pipeline.model_components.encoders import (
        Idefics3VisionEncoder,
    )
"""

from executorch.examples.qualcomm.oss_scripts.llama.model.audio_encoder import (
    GraniteSpeechCTCEncoderWrapper,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
    Idefics3VisionEncoder,
    InternVL3VisionEncoder,
)

__all__ = [
    # audio
    "GraniteSpeechCTCEncoderWrapper",
    # vision
    "Idefics3VisionEncoder",
    "InternVL3VisionEncoder",
]
