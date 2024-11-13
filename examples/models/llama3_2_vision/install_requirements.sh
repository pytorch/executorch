#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NIGHTLY_VERSION="dev20241112"

# Install torchtune nightly for model definitions.
pip install --pre torchtune==0.4.0.${NIGHTLY_VERSION} --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir

# Install torchao.
pip install "$(dirname "$0")/../../../third-party/ao"
