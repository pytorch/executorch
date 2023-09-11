#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Do shallow clone of Executorch so that we can init lintrunner in Docker build context
git clone "https://${GITHUB_TOKEN}@github.com/pytorch/executorch.git" --depth 1
chown -R ci-user executorch

pushd executorch
# Install all linter dependencies
pip_install -r requirements-lintrunner.txt
conda_run lintrunner init

# Cache .lintbin directory as part of the Docker image
cp -r .lintbin /tmp
popd

# Cleaning up
rm -rf executorch
