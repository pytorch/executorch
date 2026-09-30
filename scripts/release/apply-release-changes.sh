#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

: '
# Prepare an existing release branch. cut-release-branch.sh invokes this
# automatically; it may also be rerun safely by itself.
#
# Usage (run from root of project):
#   TEST_INFRA_BRANCH=release/2.3 ./scripts/release/apply-release-changes.sh
#
# TEST_INFRA_BRANCH: The release branch of test-infra that houses all reusable
# workflows. Defaults to the matching release/X.Y branch. The script also
# enables the PyTorch dependency for full release wheels using torch_pin.py.
# TORCH_VERSION: Optional PyTorch RC version override. By default, the newest
# release-form version available on the PyTorch test index is selected.
'

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}
RELEASE_BRANCH="release/${RELEASE_VERSION}"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}
TEST_INFRA_BRANCH=${TEST_INFRA_BRANCH:-${RELEASE_BRANCH}}

# Check out to Release Branch

if git show-ref --verify --quiet "refs/heads/${RELEASE_BRANCH}"; then
  echo "Check out local Release Branch '${RELEASE_BRANCH}'"
  git checkout "${RELEASE_BRANCH}"
elif git ls-remote --exit-code origin "${RELEASE_BRANCH}" >/dev/null 2>&1; then
  echo "Check out to Release Branch '${RELEASE_BRANCH}'"
  git checkout "${RELEASE_BRANCH}"
else
  echo "Error: Remote branch '${RELEASE_BRANCH}' not found. Please run 'cut-release-branch.sh' first."
  exit 1
fi

PREPARE_ARGS=(
  --release-version "${RELEASE_VERSION}"
  --test-infra-branch "${TEST_INFRA_BRANCH}"
)
if [[ -n "${TORCH_VERSION:-}" ]]; then
  PREPARE_ARGS+=(--torch-version "${TORCH_VERSION}")
fi
"${PYTHON_EXECUTABLE}" "${GIT_TOP_DIR}/scripts/release/prepare_release.py" "${PREPARE_ARGS[@]}"

echo "You'll need to manually commit the changes and create a PR. Here are the steps:"
echo "1. Stage the changes:"
echo "   git add .github/workflows/*.yml CONTRIBUTING.md docs/ extension/benchmark/apple/Benchmark/ backends/arm/public_api_manifests/ version.txt torch_pin.py"
echo "2. Commit the changes:"
echo "   git commit -m \"[RELEASE-ONLY CHANGES] Branch Cut for Release ${RELEASE_VERSION}\""
echo "3. After committing, create a pull request to merge the changes."
echo "4. When stable dependency artifacts are published, run:"
echo "   python scripts/release/finalize_release.py"
