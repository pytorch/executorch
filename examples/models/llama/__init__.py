# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.llm.modeling.text_decoder.decoder_model import DecoderModel


class Llama2Model(DecoderModel):
    """Llama2 model implementation that inherits from the generic DecoderModel."""

    pass


__all__ = [
    "Llama2Model",
]
