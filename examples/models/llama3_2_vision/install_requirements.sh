#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set +ex

# Install torchtune nightly for model definitions.
pip install --pre torchtune==0.6.1 --no-cache-dir
