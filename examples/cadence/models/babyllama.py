# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch

from executorch.backends.cadence.aot.export_example import export_model

from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    args = ModelArgs(
        dim=512,
        vocab_size=512,
        hidden_dim=1024,
        n_heads=8,
        # use_kv_cache=True,
        n_layers=1,
    )
    seq = 64
    b = 1
    model = Transformer(args)
    example_inputs = (torch.randint(0, 10, [b, seq], dtype=torch.int64),)

    export_model(model, example_inputs)


if __name__ == "__main__":
    main()
