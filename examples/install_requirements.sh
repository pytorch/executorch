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
pip install --force-reinstall --pre torchvision -i https://download.pytorch.org/whl/nightly/cpu
