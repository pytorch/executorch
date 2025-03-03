#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

export EXECUTORCH_ROOT="$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

pushd $EXECUTORCH_ROOT/examples/apple/coreml/llama

# Download stories llama110m artifacts
download_stories_model_artifacts

python export.py -n model.pte -p params.json -c stories110M.pt --seq_length 32 --max_seq_length 64 --dtype fp16 --coreml-quantize c4w


python run.py -m model.pte -t tokenizer.model --prompt "Once upon a time," --temperature 0.0 &> tmp.txt
tail -n +6 tmp.txt &> output.txt

cat output.txt

printf 'Once upon a time,there was a little girl named L ily . She loved to play outside in the sun sh ine . One day , she saw ' &> expected.txt


if diff output.txt expected.txt > /dev/null; then
    echo "Output matches."
else
    echo "Output does not match."
    echo "\n\nExpected:"
    cat expected.txt

    echo "\n\nGot:"
    cat output.txt

    echo "\n\nDiff:"
    diff output.txt expected.txt
    exit 1
fi

popd
