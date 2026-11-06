#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Checks build_idf_component.sh's version stamping guards in a scratch git
# repository holding only the generator and version.txt.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(mktemp -d)
trap 'rm -rf "${repo}"' EXIT
mkdir -p "${repo}/examples/espressif"
cp "${script_dir}/build_idf_component.sh" "${repo}/examples/espressif/"
echo 1.6.0 > "${repo}/version.txt"

export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com
export GIT_AUTHOR_DATE=2026-11-04T12:00:00Z GIT_COMMITTER_DATE=2026-11-04T12:00:00Z
# Signing or hooks in the developer's git config must not reach the scratch repo.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
git() { command git -C "${repo}" "$@"; }
git init -q -b main
git add -A
git commit -q -m base
before_tag=$(git rev-parse HEAD)
echo "# packaging tweak" >> "${repo}/examples/espressif/build_idf_component.sh"
git commit -q -am "packaging tweak"

failures=0
# Expects the generator to succeed and print exactly the expected output:
# description, output, arguments.
expect_version() {
    local description=$1 expected=$2 actual
    shift 2
    if actual=$(bash "${repo}/examples/espressif/build_idf_component.sh" "$@" 2>&1) \
        && [[ "${actual}" == "${expected}" ]]; then
        echo "OK: ${description}: ${expected}"
    else
        echo "FAIL: ${description}: expected ${expected}, got: ${actual}" >&2
        failures=$((failures + 1))
    fi
}
# Expects the generator to fail with a message: description, message, arguments.
expect_error() {
    local description=$1 expected=$2 actual
    shift 2
    if actual=$(bash "${repo}/examples/espressif/build_idf_component.sh" "$@" 2>&1); then
        echo "FAIL: ${description}: stamped ${actual}, expected an error with '${expected}'" >&2
        failures=$((failures + 1))
    elif [[ "${actual}" != *"${expected}"* ]]; then
        echo "FAIL: ${description}: expected '${expected}' in: ${actual}" >&2
        failures=$((failures + 1))
    else
        echo "OK: ${description}: refused with '${expected}'"
    fi
}

expect_error "release stamp off a tag" "release tag" --version --version-suffix ""
expect_version "clean needs no stamp" "Nothing to clean: ${repo}/out is not a generated component." \
    --out "${repo}/out" --clean
expect_error "executorch from another commit" "installed from this checkout" \
    --allow-dev-stamp --out "${repo}/out"
expect_error "release stamp by default" "release tag" --version
expect_error "dev suffix" "--allow-dev-stamp makes" --version --version-suffix -dev.20261104
expect_version "dev stamp from the commit date" 1.6.0-dev.20261104 --version --allow-dev-stamp
GIT_COMMITTER_DATE=2026-11-04T23:30:00-05:00 git commit -q --allow-empty -m "late commit"
expect_version "dev stamp from the commit's UTC date" 1.6.0-dev.20261105 --version --allow-dev-stamp
expect_error "unknown suffix" "must be empty, -rc.N, ~N or -dev" --version --version-suffix -beta.1
expect_version "published dev stamp" 1.6.0-dev.20261105 --version --version-suffix -dev
echo "# local edit" >> "${repo}/examples/espressif/build_idf_component.sh"
expect_error "published dev stamp with uncommitted changes" "uncommitted" \
    --version --version-suffix -dev
expect_version "unpublished dev stamp with uncommitted changes" 1.6.0-dev.20261105 \
    --version --allow-dev-stamp
git checkout -q -- .
mkdir -p "${repo}/kernels/portable/cpu"
echo "// local kernel" > "${repo}/kernels/portable/cpu/op_local.cpp"
expect_error "published dev stamp with an untracked kernel source" \
    "kernels/portable/cpu/op_local.cpp" --version --version-suffix -dev
rm -r "${repo}/kernels"
expect_error "malformed rc suffix" "-rc.N" --version --version-suffix -rc.0
expect_error "dev stamp with a suffix" "--allow-dev-stamp" --version --allow-dev-stamp --version-suffix "~1"
expect_error "malformed respin suffix" "~N with N >= 1" --version --version-suffix "~0"
expect_error "respin off a release branch" "release/1.6" --version --version-suffix "~1"
expect_error "rc off its tag" "v1.6.0-rc1" --version --version-suffix -rc.1
git branch release/1.6
expect_error "respin without its release tag" "the release tag v1.6.0" --version --version-suffix "~1"
git branch -q -D release/1.6

git tag v1.6.0-rc1
expect_version "rc stamp at its tag" 1.6.0-rc.1 --version --version-suffix -rc.1
git tag v1.6.0
expect_version "release stamp at a tag shared with an rc" 1.6.0 --version --version-suffix ""
echo "# local edit" >> "${repo}/examples/espressif/build_idf_component.sh"
expect_error "release stamp with uncommitted changes" "uncommitted" --version --version-suffix ""
git checkout -q -- .
mkdir -p "${repo}/kernels/portable/cpu"
echo "// local kernel" > "${repo}/kernels/portable/cpu/op_local.cpp"
expect_error "release stamp with an untracked kernel source" "kernels/portable/cpu/op_local.cpp" \
    --version --version-suffix ""
rm -r "${repo}/kernels"
echo scratch > "${repo}/notes.txt"
expect_version "release stamp with an untracked file the generator does not read" 1.6.0 \
    --version --version-suffix ""
rm "${repo}/notes.txt"
mkdir -p "${repo}/third-party/flatbuffers/include/flatbuffers"
echo "// base" > "${repo}/third-party/flatbuffers/include/flatbuffers/base.h"
command git -C "${repo}/third-party/flatbuffers" init -q
command git -C "${repo}/third-party/flatbuffers" add -A
command git -C "${repo}/third-party/flatbuffers" commit -q -m flatbuffers
echo "// local" > "${repo}/third-party/flatbuffers/include/flatbuffers/local.h"
expect_error "release stamp with an untracked flatbuffers header" \
    "third-party/flatbuffers/include/flatbuffers/local.h" --version --version-suffix ""
rm -rf "${repo}/third-party"
expect_version "rc stamp at a tag shared with the release" 1.6.0-rc.1 --version --version-suffix -rc.1

git checkout -q -b release/1.6
expect_error "respin at the release itself" "HEAD is the release" --version --version-suffix "~1"
echo "# packaging fix" >> "${repo}/examples/espressif/build_idf_component.sh"
git commit -q -am "packaging fix"
expect_version "respin on the release branch" "1.6.0~1" --version --version-suffix "~1"
git update-ref refs/remotes/origin/release/1.6 HEAD
git checkout -q --detach
git branch -q -D release/1.6
expect_version "respin on origin's release branch" "1.6.0~1" --version --version-suffix "~1"
git checkout -q -b release/1.6
expect_error "release stamp off the tag" "release tag" --version --version-suffix ""
mkdir -p "${repo}/.ci/scripts" "${repo}/.github/workflows" "${repo}/docs"
echo "# pipeline fix" > "${repo}/.ci/scripts/utils.sh"
echo "# pipeline fix" > "${repo}/.github/workflows/publish.yml"
echo "docs fix" > "${repo}/docs/espressif.md"
git add .ci .github docs
git commit -q -m "CI and docs fixes"
expect_version "respin with CI and docs fixes" "1.6.0~1" --version --version-suffix "~1"
git checkout -q -b side v1.6.0
echo "# side fix" >> "${repo}/examples/espressif/build_idf_component.sh"
git commit -q -am "side fix"
expect_error "respin off the release branch" "release/1.6" --version --version-suffix "~1"
git checkout -q release/1.6
echo "runtime change" > "${repo}/runtime.txt"
git add runtime.txt
git commit -q -m "runtime change"
expect_error "respin that changes the runtime" "runtime.txt" --version --version-suffix "~1"
git checkout -q --detach "${before_tag}"
expect_error "respin from before its release tag" "after the release tag" --version --version-suffix "~1"

# Expects a generator copied into a tree to refuse a dev stamp for want of
# that tree's history: description, tree.
expect_no_history() {
    local description=$1 tree=$2 actual
    mkdir -p "${tree}/examples/espressif"
    cp "${script_dir}/build_idf_component.sh" "${tree}/examples/espressif/"
    echo 1.6.0 > "${tree}/version.txt"
    if actual=$(bash "${tree}/examples/espressif/build_idf_component.sh" --version \
        --allow-dev-stamp 2>&1); then
        echo "FAIL: ${description}: stamped ${actual}" >&2
        failures=$((failures + 1))
    elif [[ "${actual}" != *"git checkout"* ]]; then
        echo "FAIL: ${description}: expected 'git checkout' in: ${actual}" >&2
        failures=$((failures + 1))
    else
        echo "OK: ${description}: refused with 'git checkout'"
    fi
}
nogit=$(mktemp -d)
trap 'rm -rf "${repo}" "${nogit}"' EXIT
GIT_CEILING_DIRECTORIES=$(dirname "${nogit}") expect_no_history "stamp without git history" \
    "${nogit}/executorch"
expect_no_history "stamp inside another repository" "${repo}/vendor/executorch"

if [[ "${failures}" -gt 0 ]]; then
    echo "FAIL: ${failures} version stamp checks failed" >&2
    exit 1
fi
echo "PASS: version stamp guards"
