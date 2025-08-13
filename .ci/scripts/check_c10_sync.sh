#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux
ls pytorch/.git || git clone https://github.com/pytorch/pytorch.git
pytorch_pin="$(< .ci/docker/ci_commit_pins/pytorch.txt)"
pushd pytorch
git checkout "$pytorch_pin"
popd
"$(dirname "${BASH_SOURCE[0]}")"/compare_dirs.sh runtime/core/portable_type/c10/c10 pytorch/c10
"$(dirname "${BASH_SOURCE[0]}")"/compare_dirs.sh runtime/core/portable_type/c10/torch/headeronly pytorch/torch/headeronly
