#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

: '
So you are looking to cut a release branch? Well you came
to the right script.

For `pytorch/executorch`, run:
> DRY_RUN=disabled ./scripts/release/cut-release-branch.sh

The script always cuts from origin/viable/strict. RELEASE_VERSION defaults to
the MAJOR.MINOR value in its version.txt (for example, 1.10.0a0
becomes release/1.10). It preserves the unmodified cut as orig/release/X.Y,
then prepares, commits, and pushes release/X.Y. TORCH_VERSION may override the
newest PyTorch release candidate selected from the test wheel index.
'

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
GIT_REMOTE=${GIT_REMOTE:-origin}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: release cuts require a clean checkout."
    exit 1
fi

(
    set -x
    git fetch --all
)

# Read version.txt directly from viable/strict so the caller's starting branch
# cannot influence which release is cut. This preserves all numeric components
# (for example, 1.10 does not become 1.1).
SOURCE_BRANCH="${GIT_REMOTE}/viable/strict"
SOURCE_VERSION=$(git show "${SOURCE_BRANCH}:version.txt")
RELEASE_VERSION=${RELEASE_VERSION:-$(printf '%s\n' "${SOURCE_VERSION}" | cut -d'.' -f1-2)}
RELEASE_BRANCH="release/${RELEASE_VERSION}"
ORIGINAL_BRANCH="orig/${RELEASE_BRANCH}"

if [[ ${DRY_RUN:-enabled} != "disabled" ]]; then
    echo "Dry run: would preserve ${SOURCE_BRANCH} as ${ORIGINAL_BRANCH}"
    echo "Dry run: would prepare, commit, and push ${RELEASE_BRANCH}"
    exit 0
fi

if ! git ls-remote --exit-code "${GIT_REMOTE}" "refs/heads/${ORIGINAL_BRANCH}" >/dev/null 2>&1; then
    git push "${GIT_REMOTE}" "${SOURCE_BRANCH}:refs/heads/${ORIGINAL_BRANCH}"
fi

if git show-ref --verify --quiet "refs/heads/${RELEASE_BRANCH}"; then
    git checkout "${RELEASE_BRANCH}"
elif git ls-remote --exit-code "${GIT_REMOTE}" "refs/heads/${RELEASE_BRANCH}" >/dev/null 2>&1; then
    git checkout -b "${RELEASE_BRANCH}" "${GIT_REMOTE}/${RELEASE_BRANCH}"
else
    git checkout -b "${RELEASE_BRANCH}" "${SOURCE_BRANCH}"
fi

(
    set -x
    RELEASE_VERSION="${RELEASE_VERSION}" \
      "${GIT_TOP_DIR}/scripts/release/apply-release-changes.sh"
)

git add -A
if git diff --cached --quiet; then
    echo "Release branch is already prepared; nothing to commit."
else
    git commit -m "[RELEASE-ONLY CHANGES] Branch Cut for Release ${RELEASE_VERSION}"
fi
git push --set-upstream "${GIT_REMOTE}" "${RELEASE_BRANCH}"
