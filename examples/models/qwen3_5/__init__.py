# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.qwen3_5.convert_weights import convert_weights

__all__ = ["Qwen3_5Model", "convert_weights"]


def __getattr__(name):
    if name == "Qwen3_5Model":
        from executorch.examples.models.llama.model import Llama2Model

        class Qwen3_5Model(Llama2Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        globals()["Qwen3_5Model"] = Qwen3_5Model
        return Qwen3_5Model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
