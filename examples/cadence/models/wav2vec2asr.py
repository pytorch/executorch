# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

from ..aot.meta_registrations import *  # noqa

import torchaudio

from torchaudio.utils import download_asset

from ..aot.export_example import export_model


if __name__ == "__main__":

    SPEECH_FILE = download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    )

    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE

    model = bundle.get_model()

    labels = bundle.get_labels()

    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to("cpu")

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate
        )
    example_inputs = (waveform,)

    export_model(model, example_inputs)
