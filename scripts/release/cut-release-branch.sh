#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

: '
So you are looking to cut a release branch? Well you came
to the right script.

This script can be used to cut any branch on any repository

For `pytorch/executorch` usage would be like:
> DRY_RUN=disabled ./scripts/release/cut-release-branch.sh

or to cut from main branch:
> DRY_RUN=disabled GIT_BRANCH_TO_CUT_FROM=main ./scripts/release/cut-release-branch.sh

After creating release/X.Y and orig/release/X.Y at the selected commit, this
also applies the deterministic release-only changes to the local release/X.Y
branch. TEST_INFRA_BRANCH defaults to release/X.Y. TORCH_VERSION may override
the newest PyTorch release candidate selected from the test wheel index.
'

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
GIT_REMOTE=${GIT_REMOTE:-origin}
GIT_BRANCH_TO_CUT_FROM=${GIT_BRANCH_TO_CUT_FROM:-viable/strict}

# should output something like 1.11
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}
TEST_INFRA_BRANCH=${TEST_INFRA_BRANCH:-release/${RELEASE_VERSION}}

DRY_RUN_FLAG="--dry-run"
if [[ ${DRY_RUN:-enabled} == "disabled" ]]; then
    DRY_RUN_FLAG=""
fi


(
    set -x
    git fetch --all
    git checkout "${GIT_REMOTE}/${GIT_BRANCH_TO_CUT_FROM}"
)

for branch in "release/${RELEASE_VERSION}" "orig/release/${RELEASE_VERSION}"; do
    if git rev-parse --verify "${branch}" >/dev/null 2>/dev/null; then
        echo "+ Branch ${branch} already exists, skipping..."
        continue
    else
        (
            set -x
            git checkout "${GIT_REMOTE}/${GIT_BRANCH_TO_CUT_FROM}"
            git checkout -b "${branch}"
            git push -q ${DRY_RUN_FLAG} "${GIT_REMOTE}" "${branch}"
        )
    fi
done

(
    set -x
    RELEASE_VERSION="${RELEASE_VERSION}" \
      TEST_INFRA_BRANCH="${TEST_INFRA_BRANCH}" \
      "${GIT_TOP_DIR}/scripts/release/apply-release-changes.sh"
)
