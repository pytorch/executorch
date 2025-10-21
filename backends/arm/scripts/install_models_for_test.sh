#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e
pip install -r backends/arm/requirements-arm-models-test.txt

# Install model gym repository
git clone https://github.com/arm/neural-graphics-model-gym.git
cd neural-graphics-model-gym
# Remove model-converter installation from model-gym repository (to prevent overwriting executorch version)
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' 's/^model-converter  = "ng_model_gym.bin.model_converter_launcher:main"/#&/' pyproject.toml
else
    sed -i 's/^model-converter  = "ng_model_gym.bin.model_converter_launcher:main"/#&/' pyproject.toml
fi
pip install . --no-deps
cd ..
rm -rf neural-graphics-model-gym