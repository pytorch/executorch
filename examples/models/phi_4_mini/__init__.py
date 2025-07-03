# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.phi_4_mini.convert_weights import convert_weights
from executorch.extension.llm.modeling.text_decoder.decoder_model import DecoderModel


class Phi4MiniModel(DecoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "Phi4MiniModel",
    "convert_weights",
]
