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

REPORT_FILE="$ARTIFACT_DIR/test-report-$FLOW-$SUITE.csv"

echo "Running backend test job for suite $SUITE, flow $FLOW."
echo "Saving job artifacts to $ARTIFACT_DIR."

${CONDA_RUN} --no-capture-output pip install awscli==1.37.21

bash .ci/scripts/setup-conda.sh
eval "$(conda shell.bash hook)"

PYTHON_EXECUTABLE=python
${CONDA_RUN} --no-capture-output .ci/scripts/setup-macos.sh --build-tool cmake --build-mode Release

EXIT_CODE=0
pytest -c /dev/nul -n auto backends/test/suite/$SUITE/ -m flow_$FLOW --json-report "$REPORT_FILE" || EXIT_CODE=$?

# Generate markdown summary.
python -m executorch.backends.test.suite.generate_markdown_summary_json "$REPORT_FILE" > ${GITHUB_STEP_SUMMARY:-"step_summary.md"} --exit-code $EXIT_CODE
