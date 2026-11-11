#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Checks build_idf_component.sh's version stamping guards in a scratch git
# repository holding only the generator, gen_kernels.py, version.txt, the
# install_requirements.py it reads the torch pin from, a FlatBuffers
# submodule without its sources, and a README the generator does not read.
# Needs git and a Python with PyYAML, which gen_kernels.py imports.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(mktemp -d) nogit=$(mktemp -d) superproject=$(mktemp -d) flatbuffers=$(mktemp -d)
packages=$(mktemp -d)
trap 'rm -rf "${repo}" "${nogit}" "${superproject}" "${flatbuffers}" "${packages}"' EXIT
mkdir -p "${repo}/examples/espressif"
cp "${script_dir}/build_idf_component.sh" "${repo}/examples/espressif/"
mkdir -p "${repo}/examples/espressif/idf_component_template/tools"
cp "${script_dir}/idf_component_template/tools/gen_kernels.py" \
    "${repo}/examples/espressif/idf_component_template/tools/"
echo 1.6.0 > "${repo}/version.txt"
cp "${script_dir}/../../install_requirements.py" "${script_dir}/../../install_utils.py" "${repo}/"
echo readme > "${repo}/README.md"

export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com
export GIT_AUTHOR_DATE=2026-11-04T12:00:00Z GIT_COMMITTER_DATE=2026-11-04T12:00:00Z
# Signing or hooks in the developer's git config must not reach the scratch repo.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
git() { command git -C "${repo}" "$@"; }
# FlatBuffers' headers at two commits, the submodule pinned at the second.
mkdir -p "${flatbuffers}/include/flatbuffers"
echo "// base" > "${flatbuffers}/include/flatbuffers/base.h"
command git -C "${flatbuffers}" init -q -b main
command git -C "${flatbuffers}" add -A
command git -C "${flatbuffers}" commit -q -m one
echo "// base, revised" > "${flatbuffers}/include/flatbuffers/base.h"
command git -C "${flatbuffers}" commit -q -am two
git init -q -b main
git -c protocol.file.allow=always submodule add -q "${flatbuffers}" third-party/flatbuffers
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
    if actual=$(bash "${generator:-${repo}/examples/espressif/build_idf_component.sh}" "$@" 2>&1) \
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
    if actual=$(bash "${generator:-${repo}/examples/espressif/build_idf_component.sh}" "$@" 2>&1); then
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
pin=$(<"${repo}/install_requirements.py")
printf '%s\n' "${pin//\"torch==/\"torch>=}" > "${repo}/install_requirements.py"
expect_error "a torch pin the generator cannot read" "does not name one torch==X" \
    --version --allow-dev-stamp
git checkout -q -- install_requirements.py
echo "import no_such_module" >> "${repo}/install_utils.py"
expect_error "an install_requirements.py that does not import" "No module named 'no_such_module'" \
    --version --allow-dev-stamp
git checkout -q -- install_utils.py
expect_version "clean needs no stamp" "Nothing to clean: ${repo}/out is not a generated component." \
    --out "${repo}/out" --clean
expect_error "release stamp by default" "release tag" --version
expect_error "dev suffix" "--allow-dev-stamp makes" --version --version-suffix -dev.20261104
expect_version "dev stamp from the commit date" 1.6.0-dev.20261104 --version --allow-dev-stamp
GIT_COMMITTER_DATE=2026-11-04T23:30:00-05:00 git commit -q --allow-empty -m "late commit"
expect_version "dev stamp from the commit's UTC date" 1.6.0-dev.20261105 --version --allow-dev-stamp
expect_error "unknown suffix" "must be empty, -rc.N, ~N or -dev" --version --version-suffix -beta.1
expect_error "published dev stamp off the nightly branch" "nightly branch" \
    --version --version-suffix -dev
git branch nightly
expect_version "published dev stamp on the nightly branch" 1.6.0-dev.20261105 \
    --version --version-suffix -dev
git update-ref refs/remotes/origin/nightly nightly
git branch -q -D nightly
expect_version "published dev stamp on origin's nightly branch" 1.6.0-dev.20261105 \
    --version --version-suffix -dev
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
expect_error "rc off its tag" "v1.6.0-rc1" --version --version-suffix -rc.1
git tag v1.6.0-esp-idf.1
expect_error "respin off a release branch" "release/1.6" --version --version-suffix "~1"
git branch release/1.6
expect_error "respin without its release tag" "needs the release tag" --version --version-suffix "~1"
git branch -q -D release/1.6
git tag -d v1.6.0-esp-idf.1 > /dev/null

git tag v1.6.0-rc1
expect_version "rc stamp at its tag" 1.6.0-rc.1 --version --version-suffix -rc.1
echo "# local edit" >> "${repo}/examples/espressif/build_idf_component.sh"
expect_error "rc stamp with uncommitted changes" "uncommitted" --version --version-suffix -rc.1
git checkout -q -- .
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
echo '*.swp' > "${repo}/.gitignore"
mkdir -p "${repo}/codegen/templates"
echo swap > "${repo}/codegen/templates/RegisterKernels.cpp.swp"
expect_error "release stamp with an ignored file in a tree it copies" \
    "codegen/templates/RegisterKernels.cpp.swp" --version --version-suffix ""
rm -r "${repo}/codegen" "${repo}/.gitignore"
echo scratch > "${repo}/notes.txt"
expect_version "release stamp with an untracked file the generator does not read" 1.6.0 \
    --version --version-suffix ""
rm "${repo}/notes.txt"
echo "local edit" >> "${repo}/README.md"
expect_version "release stamp with a change the generator does not read" 1.6.0 \
    --version --version-suffix ""
git checkout -q -- README.md
submodule="${repo}/third-party/flatbuffers"
echo "// local" > "${submodule}/include/flatbuffers/local.h"
expect_error "release stamp with an untracked flatbuffers header" \
    "third-party/flatbuffers/include/flatbuffers/local.h" --version --version-suffix ""
rm "${submodule}/include/flatbuffers/local.h"
echo "// local edit" >> "${submodule}/include/flatbuffers/base.h"
expect_error "release stamp with an edited flatbuffers header" "M third-party/flatbuffers" \
    --version --version-suffix ""
command git -C "${submodule}" checkout -q -- .
command git -C "${submodule}" checkout -q HEAD~1
expect_error "release stamp with flatbuffers at another commit" "M third-party/flatbuffers" \
    --version --version-suffix ""
command git -C "${submodule}" checkout -q main
expect_version "rc stamp at a tag shared with the release" 1.6.0-rc.1 --version --version-suffix -rc.1

git checkout -q -b release/1.6
git tag v1.6.0-esp-idf.1
expect_error "respin at the release itself" "HEAD is the release" --version --version-suffix "~1"
git tag -d v1.6.0-esp-idf.1 > /dev/null
echo "# packaging fix" >> "${repo}/examples/espressif/build_idf_component.sh"
git commit -q -am "packaging fix"
expect_error "respin without its tag" "v1.6.0-esp-idf.1" --version --version-suffix "~1"
git tag v1.6.0-esp-idf.1
expect_version "respin at its tag on the release branch" "1.6.0~1" --version --version-suffix "~1"
expect_error "respin at another respin's tag" "v1.6.0-esp-idf.2" --version --version-suffix "~2"
echo "# local edit" >> "${repo}/examples/espressif/build_idf_component.sh"
expect_error "respin with uncommitted changes" "uncommitted" --version --version-suffix "~1"
git checkout -q -- .
# Packages past whose check the generator stops at the empty FlatBuffers
# submodule: the release's, and a source build of another commit.
mkdir -p "${packages}/release/executorch" "${packages}/other/executorch"
touch "${packages}/release/executorch/__init__.py" "${packages}/other/executorch/__init__.py"
printf '__version__ = "1.6.0"\ngit_version = "%s"\n' "$(git rev-parse v1.6.0)" \
    > "${packages}/release/executorch/version.py"
printf '__version__ = "1.6.0+%s"\ngit_version = "%s"\n' "${before_tag:0:7}" "${before_tag}" \
    > "${packages}/other/executorch/version.py"
PYTHONPATH="${packages}/release" expect_error "respin with the release's package" \
    "third-party/flatbuffers is empty" --version-suffix "~1" --out "${repo}/out"
PYTHONPATH="${packages}/other" expect_error "respin with a package from another commit" \
    "found 1.6.0+${before_tag:0:7} at ${before_tag}" --version-suffix "~1" --out "${repo}/out"
PYTHONPATH="${packages}/other" expect_error "dev stamp with a package from another commit" \
    "leaves out examples/hello_executorch" --allow-dev-stamp --out "${repo}/out"
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
expect_error "a later commit reusing a respin's number" "v1.6.0-esp-idf.1" \
    --version --version-suffix "~1"
git tag v1.6.0-esp-idf.2
expect_version "respin with CI and docs fixes" "1.6.0~2" --version --version-suffix "~2"
git checkout -q -b side v1.6.0
echo "# side fix" >> "${repo}/examples/espressif/build_idf_component.sh"
git commit -q -am "side fix"
git tag v1.6.0-esp-idf.3
expect_error "respin off the release branch" "release/1.6" --version --version-suffix "~3"
git checkout -q release/1.6
echo "runtime change" > "${repo}/runtime.txt"
git add runtime.txt
git commit -q -m "runtime change"
git tag v1.6.0-esp-idf.4
expect_error "respin that changes the runtime" "runtime.txt" --version --version-suffix "~4"
git checkout -q --detach "${before_tag}"
git tag v1.6.0-esp-idf.5
expect_error "respin from before its release tag" "after the release tag" \
    --version --version-suffix "~5"

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
GIT_CEILING_DIRECTORIES=$(dirname "${nogit}") expect_no_history "stamp without git history" \
    "${nogit}/executorch"
expect_no_history "stamp inside another repository" "${repo}/vendor/executorch"
command git -C "${superproject}" init -q -b main
command git -C "${superproject}" -c protocol.file.allow=always submodule add -q -b release/1.6 \
    "${repo}" vendor/executorch
generator="${superproject}/vendor/executorch/examples/espressif/build_idf_component.sh" \
    expect_version "stamp in a submodule of another repository" 1.6.0-dev.20261104 \
    --version --allow-dev-stamp

if [[ "${failures}" -gt 0 ]]; then
    echo "FAIL: ${failures} version stamp checks failed" >&2
    exit 1
fi
echo "PASS: version stamp guards"
