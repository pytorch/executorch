#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

: '
# Step 2 after branch cut is complete.
#
# Creates PR with release only changes.
#
# Usage (run from root of project):
#   TEST_INFRA_BRANCH=release/2.3 ./scripts/release/apply-release-changes.sh
#
# TEST_INFRA_BRANCH: The release branch of test-infra that houses all reusable
'

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}
RELEASE_BRANCH="release/${RELEASE_VERSION}"

# Check out to Release Branch

if git ls-remote --exit-code origin ${RELEASE_BRANCH} >/dev/null 2>&1; then
  echo "Check out to Release Branch '${RELEASE_BRANCH}'"
  git checkout ${RELEASE_BRANCH}
else
  echo "Error: Remote branch '${RELEASE_BRANCH}' not found. Please run 'cut-release-branch.sh' first."
  exit 1
fi

# Change all GitHub Actions to reference the test-infra release branch
# as opposed to main.
echo "Applying release-only changes to workflows"
for i in .github/workflows/*.yml; do
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e s#@main#@"${TEST_INFRA_BRANCH}"# $i;
    sed -i '' -e s#test-infra-ref:[[:space:]]main#"test-infra-ref: ${TEST_INFRA_BRANCH}"# $i;
  else
    sed -i -e s#@main#@"${TEST_INFRA_BRANCH}"# $i;
    sed -i -e s#test-infra-ref:[[:space:]]main#"test-infra-ref: ${TEST_INFRA_BRANCH}"# $i;
  fi
done

echo "You'll need to manually commit the changes and create a PR. Here are the steps:"
echo "1. Stage the changes to the workflow files:"
echo "   git add ./github/workflows/*.yml"
echo "2. Commit the changes:"
echo "   git commit -m \"[RELEASE-ONLY CHANGES] Branch Cut for Release ${RELEASE_VERSION}\""
echo "3. After committing, create a pull request to merge the changes."
