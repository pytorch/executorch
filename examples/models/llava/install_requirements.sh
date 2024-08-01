#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x
OS=$(uname)

# install llava from the submodule. We can't do pip install llava because it is packaged incorrectly.
if [[ $OS != "Darwin" ]];
then
    #This doesn't work for macos, on python 3.12, because torch 2.1.2 is missing.
    pip install --force-reinstall -e examples/third-party/LLaVA
else
    # manually install dependencies
    pip install tokenizers==0.15.1 sentencepiece==0.1.99    \
        shortuuid accelerate==0.21.0 peft                   \
        pydantic markdown2[all]  scikit-learn==1.2.2        \
        requests httpx==0.24.0 uvicorn fastapi              \
        einops==0.6.1 einops-exts==0.0.4 timm==0.6.13

    pip install --force-reinstall -e examples/third-party/LLaVA --no-deps
fi

# not included in the pip install package, but needed in llava
pip install protobuf
pip install triton==3.0.0

# The deps of llava can have different versions than deps of ExecuTorch.
# For example, torch version required from llava is older than ExecuTorch.
# To make both work, recover ExecuTorch's original dependencies by rerunning
# the install_requirements.sh. Notice this won't install executorch.
bash -x ./install_requirements.sh --pybind xnnpack
