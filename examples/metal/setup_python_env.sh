#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Check if we're in the executorch directory
if [ "$(basename "$(pwd)")" != "executorch" ]; then
  echo "Error: Not in executorch directory"
  echo "Current directory: $(pwd)"
  echo "Please navigate to the executorch directory"
  exit 1
fi

if [ ! -f "pyproject.toml" ] || [ ! -d "backends" ] || [ ! -f "install_executorch.sh" ]; then
  echo "Error: executorch directory structure is incomplete"
  exit 1
fi

# Install Optimum-ExecuTorch
export OPTIMUM_ET_VERSION=$(cat .ci/docker/ci_commit_pins/optimum-executorch.txt)

echo "Installing Optimum-ExecuTorch version: $OPTIMUM_ET_VERSION"
pip install git+https://github.com/huggingface/optimum-executorch.git@${OPTIMUM_ET_VERSION}

# Install required libraries
echo "Installing required libraries"
pip install mistral-common librosa accelerate

# Install ExecuTorch
echo "Installing ExecuTorch"
./install_executorch.sh
