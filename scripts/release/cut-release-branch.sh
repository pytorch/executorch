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

VIABLE_BRANCH="${GIT_REMOTE}/viable/strict"
if [[ -z "${RELEASE_VERSION:-}" ]]; then
    # A failed first run may already have preserved the cut point. Resume that
    # release even if viable/strict has since moved to the next line.
    PENDING_RELEASES=()
    while IFS= read -r ORIGINAL_REF; do
        CANDIDATE=${ORIGINAL_REF#refs/remotes/${GIT_REMOTE}/orig/release/}
        if ! git show-ref --verify --quiet "refs/remotes/${GIT_REMOTE}/release/${CANDIDATE}"; then
            PENDING_RELEASES+=("${CANDIDATE}")
        fi
    done < <(git for-each-ref --format='%(refname)' "refs/remotes/${GIT_REMOTE}/orig/release/")
    if [[ ${#PENDING_RELEASES[@]} -gt 1 ]]; then
        echo "Error: multiple unfinished release cuts: ${PENDING_RELEASES[*]}"
        exit 1
    elif [[ ${#PENDING_RELEASES[@]} -eq 1 ]]; then
        RELEASE_VERSION="${PENDING_RELEASES[0]}"
    else
        # Read version.txt directly from viable/strict so the caller's starting
        # branch cannot influence a new release cut.
        SOURCE_VERSION=$(git show "${VIABLE_BRANCH}:version.txt")
        RELEASE_VERSION=$(printf '%s\n' "${SOURCE_VERSION}" | cut -d'.' -f1-2)
    fi
fi
RELEASE_BRANCH="release/${RELEASE_VERSION}"
ORIGINAL_BRANCH="orig/${RELEASE_BRANCH}"
ARM_MANIFEST="backends/arm/public_api_manifests/api_manifest_${RELEASE_VERSION//./_}.toml"

# The API snapshot protects main against backwards-incompatible changes, so it
# must be reviewed and merged there before the branch is cut. Creating it only
# on the release branch leaves main unprotected and guarantees later conflicts.
if git cat-file -e "${VIABLE_BRANCH}:${ARM_MANIFEST}" 2>/dev/null; then
    :
elif git cat-file -e "${VIABLE_BRANCH}:backends/arm/public_api_manifests/api_manifest_running.toml" 2>/dev/null; then
    echo "Error: ${ARM_MANIFEST} must be merged into ${VIABLE_BRANCH} before the release cut."
    exit 1
fi

if [[ ${DRY_RUN:-enabled} != "disabled" ]]; then
    echo "Dry run: would preserve ${VIABLE_BRANCH} as ${ORIGINAL_BRANCH}"
    echo "Dry run: would prepare, commit, and push ${RELEASE_BRANCH}"
    exit 0
fi

if git ls-remote --exit-code "${GIT_REMOTE}" "refs/heads/${ORIGINAL_BRANCH}" >/dev/null 2>&1; then
    # A previous attempt already fixed the cut point. Always resume from it,
    # even if viable/strict advanced after that attempt failed.
    CUT_SOURCE="${GIT_REMOTE}/${ORIGINAL_BRANCH}"
else
    git push "${GIT_REMOTE}" "${VIABLE_BRANCH}:refs/heads/${ORIGINAL_BRANCH}"
    git fetch "${GIT_REMOTE}" "${ORIGINAL_BRANCH}:refs/remotes/${GIT_REMOTE}/${ORIGINAL_BRANCH}"
    CUT_SOURCE="${GIT_REMOTE}/${ORIGINAL_BRANCH}"
fi

if git show-ref --verify --quiet "refs/heads/${RELEASE_BRANCH}"; then
    if ! git merge-base --is-ancestor "${CUT_SOURCE}" "refs/heads/${RELEASE_BRANCH}"; then
        echo "Error: local ${RELEASE_BRANCH} does not descend from preserved cut ${CUT_SOURCE}."
        exit 1
    fi
    git checkout "${RELEASE_BRANCH}"
elif git ls-remote --exit-code "${GIT_REMOTE}" "refs/heads/${RELEASE_BRANCH}" >/dev/null 2>&1; then
    if ! git merge-base --is-ancestor "${CUT_SOURCE}" "${GIT_REMOTE}/${RELEASE_BRANCH}"; then
        echo "Error: remote ${RELEASE_BRANCH} does not descend from preserved cut ${CUT_SOURCE}."
        exit 1
    fi
    git checkout -b "${RELEASE_BRANCH}" "${GIT_REMOTE}/${RELEASE_BRANCH}"
else
    git checkout -b "${RELEASE_BRANCH}" "${CUT_SOURCE}"
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
