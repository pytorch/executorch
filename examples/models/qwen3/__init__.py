# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.qwen3.convert_weights import convert_weights
from executorch.extension.llm.modeling.text_decoder.decoder_model import DecoderModel


class Qwen3Model(DecoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "Qwen3Model",
    "convert_weights",
]
