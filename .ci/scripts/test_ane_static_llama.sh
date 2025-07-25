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

# Update tokenizers submodule
pushd $EXECUTORCH_ROOT/extension/llm/tokenizers
echo "Update tokenizers submodule"
git submodule update --init
popd

pushd $EXECUTORCH_ROOT/examples/apple/coreml/llama

# Download stories llama110m artifacts
download_stories_model_artifacts

python export.py -n model.pte -p params.json -c stories110M.pt --seq_length 32 --max_seq_length 64 --dtype fp16 --coreml-quantize c4w --embedding-quantize 4,32

popd
