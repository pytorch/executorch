#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
ls pytorch/.git || git clone https://github.com/pytorch/pytorch.git
pushd pytorch
git checkout "$(< ../.ci/docker/ci_commit_pins/pytorch.txt)"
popd
"$(dirname "${BASH_SOURCE[0]}")"/diff_c10_mirror_with_pytorch.sh
