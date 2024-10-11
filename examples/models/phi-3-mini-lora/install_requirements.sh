#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

pip install torchvision
pip install torchtune
pip install tiktoken

# Install torchao.
TORCHAO_VERSION=$(cat "$(dirname "$0")"/../../../.ci/docker/ci_commit_pins/torchao.txt)
pip install --no-use-pep517 "git+https://github.com/pytorch/ao.git@${TORCHAO_VERSION}"
