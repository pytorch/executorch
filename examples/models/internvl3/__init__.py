# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.internvl3.convert_weights import convert_weights
from executorch.examples.models.llama.model import Llama2Model


class InternVL3Model(Llama2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "InternVL3Model",
    "convert_weights",
]
