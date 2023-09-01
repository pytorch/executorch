#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# install pre-requisite
# here it is used to install torchvision's nighlty package because the latest
# variant install an older version of pytorch, 1.8,
# tested only on linux

# Note:
# When getting a new version of the executorch repo (via clone, fetch, or pull),
# you may need to re-install a new version for all dependencies in order to the
# models in executorch/examples/models.
# The version in this file will be the correct version for the
# corresponsing version of the repo.

TORCH_VISION_VERSION=0.16.0.dev20230831
pip install --force-reinstall --pre torchvision=="${TORCH_VISION_VERSION}" -i https://download.pytorch.org/whl/nightly/cpu

TORCH_AUDIO_VERSION=2.2.0.dev20230831
pip install --force-reinstall --pre torchaudio=="${TORCH_AUDIO_VERSION}" -i https://download.pytorch.org/whl/nightly/cpu

TIMM_VERSION=0.6.13
pip install --pre timm==${TIMM_VERSION}

TRANSFORMERS_VERSION=4.31.0
pip install --pre transformers==${TRANSFORMERS_VERSION}
