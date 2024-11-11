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
pip install "$(dirname "$0")/../../../third-party/ao"
