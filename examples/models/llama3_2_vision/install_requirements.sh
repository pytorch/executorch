#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install torchtune nightly for model definitions.
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir

# Install torchao.
TORCHAO_VERSION=$(cat "$(dirname "$0")"/../../../.ci/docker/ci_commit_pins/torchao.txt)
pip install --no-use-pep517 "git+https://github.com/pytorch/ao.git@${TORCHAO_VERSION}"
