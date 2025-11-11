#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

DIR_NAME="${1:-voxtral}"

mkdir -p "$DIR_NAME"

optimum-cli export executorch \
            --model "mistralai/Voxtral-Mini-3B-2507" \
            --task "multimodal-text-to-text" \
            --recipe "metal" \
            --dtype bfloat16 \
            --max_seq_len 1024 \
            --output_dir "$DIR_NAME"

python -m executorch.extension.audio.mel_spectrogram \
            --feature_size 128 \
            --stack_output \
            --max_audio_len 300 \
            --output_file "$DIR_NAME"/voxtral_preprocessor.pte

curl -L https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/resolve/main/tekken.json -o "$DIR_NAME"/tekken.json
