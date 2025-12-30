# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torchaudio

from executorch.backends.cadence.aot.export_example import export_model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    text = "Hello world! T T S stands for Text to Speech!"

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()
    model = bundle.get_tacotron2()
    model.eval()

    tokens, tokens_length = processor(text)
    specgram, specgram_lengths, _ = model.infer(tokens, tokens_length)

    inputs = (
        tokens,
        tokens_length,
        specgram,
        specgram_lengths,
    )

    # Full model
    export_model(model, inputs)


if __name__ == "__main__":
    main()
