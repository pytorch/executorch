#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reuse the runner's read-only repositories while allowing missing models to download.
HF_SHARED_HUB_CACHE="$(python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')"
export HF_HOME="$(mktemp -d "${TMPDIR:-/tmp}/executorch-hf-cache.XXXXXX")"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}"
for model_dir in "${HF_SHARED_HUB_CACHE}"/models--*; do
    if [[ -d "${model_dir}" ]]; then
        ln -s "${model_dir}" "${HF_HUB_CACHE}/"
    fi
done
