#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# WARNING: Only run this after root level install_requirements.sh!
echo "WARNING: Only run this after root level install_requirements.sh!"
set -x

# Manually install llava dependencies because torch
# Newer transformer will give TypeError: LlavaLlamaForCausalLM.forward() got an unexpected keyword argument 'cache_position'
pip install transformers==4.37.2
pip install timm==0.6.13

# not included in the pip install package, but needed in llava
pip install protobuf
pip install sentencepiece
pip install accelerate

# bitsandbytes depends on numpy 1.x, which is not compatible with numpy 2.x.
# Reinstall bitsandbytes to make it compatible. Do not install deps because it messes up torch version.
pip install bitsandbytes -I --no-deps

# numpy needs to be pin to 1.24. 1.26.4 will error out
pip install numpy==1.24

# install llava from the submodule. Do not install deps because it messes up torch version.
pip install --force-reinstall -e examples/third-party/LLaVA --no-deps
