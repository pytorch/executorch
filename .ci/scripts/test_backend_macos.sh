#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SUITE=$1
FLOW=$2
ARTIFACT_DIR=$3

echo "Running backend test job for suite $SUITE, flow $FLOW."
echo "Saving job artifacts to $ARTIFACT_DIR."

${CONDA_RUN} --no-capture-output pip install awscli==1.37.21

bash .ci/scripts/setup-conda.sh
eval "$(conda shell.bash hook)"

PYTHON_EXECUTABLE=python
${CONDA_RUN} --no-capture-output .ci/scripts/setup-macos.sh --build-tool cmake --build-mode Release

${CONDA_RUN} --no-capture-output python -m executorch.backends.test.suite.runner $SUITE --flow $FLOW --report "$ARTIFACT_DIR/test_results.csv"
