#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

# install llava from the submodule
# TODO: This doesn't work for macos, on python 3.12, because torch 2.1.2 is missing.
# manually install dependencies so we don't have conflicts on torch
pip install tokenizers==0.15.1 sentencepiece==0.1.99    \
    shortuuid accelerate==0.21.0 peft                   \
    pydantic markdown2[all]  scikit-learn==1.2.2        \
    requests httpx==0.24.0 uvicorn fastapi              \
    einops==0.6.1 einops-exts==0.0.4 timm==0.6.13

pip install --force-reinstall -e examples/third-party/LLaVA --no-deps

# not included in the pip install package, but needed in llava
pip install protobuf

# bitsandbytes depends on numpy 1.x, which is not compatible with numpy 2.x.
# Reinstall bitsandbytes to make it compatible.
pip install bitsandbytes -I

OS=$(uname)
if [[ $OS ~= "Darwin" ]];
then
    # numpy needs to be pin to 1.24. 1.26.4 will error out: Could not infer dtype of numpy.uint8
    # On macos, numpy 1.24.4 is not available, 1.26.4 works fine.
    pip install numpy==1.24.4
fi

# The deps of llava can have different versions than deps of ExecuTorch.
# For example, torch version required from llava is older than ExecuTorch.
# To make both work, recover ExecuTorch's original dependencies by rerunning
# the install_requirements.sh. Notice this won't install executorch.
bash -x ./install_requirements.sh --deps-only

# Newer transformer will give TypeError: LlavaLlamaForCausalLM.forward() got an unexpected keyword argument 'cache_position'
pip install transformers==4.37.2
