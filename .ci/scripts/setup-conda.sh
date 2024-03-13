#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_conda() {
  pushd .ci/docker || return
  ${CONDA_INSTALL} -y --file conda-env-ci.txt
  popd || return
}

install_conda
