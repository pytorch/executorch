#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../../.." rev-parse --show-toplevel)"
mlx_git="$repo_root/backends/mlx/third-party/mlx"
apply_script="$script_dir/apply.sh"
old_sdk="$script_dir/mlx_metal_sdk_per_platform_legacy.patch"
current_sdk="$script_dir/mlx_metal_sdk_per_platform.patch"
obsolete_swift="$script_dir/mlx_swiftpm_metallib_name_legacy.patch"
addrspace="$script_dir/mlx_metal_remove_addrspace_compat.patch"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/mlx-patch-test.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

create_fixture() {
  local fixture="$1"
  mkdir -p "$fixture/mlx/backend/metal/kernels/steel/utils"
  git -C "$mlx_git" show HEAD:mlx/backend/metal/kernels/CMakeLists.txt \
    > "$fixture/mlx/backend/metal/kernels/CMakeLists.txt"
  git -C "$mlx_git" show HEAD:mlx/backend/metal/device.cpp \
    > "$fixture/mlx/backend/metal/device.cpp"
  git -C "$mlx_git" show HEAD:mlx/backend/metal/kernels/steel/utils/integral_constant.h \
    > "$fixture/mlx/backend/metal/kernels/steel/utils/integral_constant.h"
  git -C "$fixture" init -q
  git -C "$fixture" add .
  git -C "$fixture" -c user.name=mlx-patch-test \
    -c user.email=mlx-patch-test@example.com commit -qm baseline
}

assert_applied() {
  local fixture="$1"
  local patch="$2"
  git -C "$fixture" apply --reverse --check "$patch" >/dev/null 2>&1 ||
    fail "patch is not exactly applied: $patch"
}

assert_absent() {
  local fixture="$1"
  local patch="$2"
  git -C "$fixture" apply --check "$patch" >/dev/null 2>&1 ||
    fail "patch is not exactly absent: $patch"
}

run_migration() {
  local fixture="$1"
  "$apply_script" "$fixture" \
    --migrate "$old_sdk" "$current_sdk" \
    --remove "$obsolete_swift" \
    --apply "$addrspace"
}

clean_fixture="$tmp_dir/clean"
create_fixture "$clean_fixture"
run_migration "$clean_fixture"
assert_applied "$clean_fixture" "$current_sdk"
assert_absent "$clean_fixture" "$obsolete_swift"
assert_applied "$clean_fixture" "$addrspace"
git -C "$clean_fixture" diff --binary > "$tmp_dir/clean-before-rerun.diff"
run_migration "$clean_fixture"
git -C "$clean_fixture" diff --binary > "$tmp_dir/clean-after-rerun.diff"
cmp -s "$tmp_dir/clean-before-rerun.diff" "$tmp_dir/clean-after-rerun.diff" ||
  fail "rerunning migration changed the working tree"
echo "PASS: clean -> current and rerun no-op"

legacy_fixture="$tmp_dir/legacy"
create_fixture "$legacy_fixture"
git -C "$legacy_fixture" apply "$old_sdk"
git -C "$legacy_fixture" apply "$obsolete_swift"
run_migration "$legacy_fixture"
assert_applied "$legacy_fixture" "$current_sdk"
assert_absent "$legacy_fixture" "$obsolete_swift"
assert_applied "$legacy_fixture" "$addrspace"
echo "PASS: legacy SDK -> current SDK and obsolete SwiftPM removal"

current_fixture="$tmp_dir/current"
create_fixture "$current_fixture"
git -C "$current_fixture" apply "$current_sdk"
git -C "$current_fixture" apply "$addrspace"
git -C "$current_fixture" diff --binary > "$tmp_dir/current-before.diff"
# Bare patch arguments are the original caller interface.
"$apply_script" "$current_fixture" "$current_sdk" "$addrspace"
git -C "$current_fixture" diff --binary > "$tmp_dir/current-after.diff"
cmp -s "$tmp_dir/current-before.diff" "$tmp_dir/current-after.diff" ||
  fail "already-current invocation changed the working tree"
echo "PASS: current state and legacy caller interface are no-ops"

relative_fixture="$tmp_dir/relative"
create_fixture "$relative_fixture"
(
  cd "$repo_root"
  backends/mlx/patches/apply.sh "$relative_fixture" \
    backends/mlx/patches/mlx_metal_sdk_per_platform.patch \
    backends/mlx/patches/mlx_metal_remove_addrspace_compat.patch
)
assert_applied "$relative_fixture" "$current_sdk"
assert_applied "$relative_fixture" "$addrspace"
echo "PASS: relative patch paths are normalized before git -C"

unrelated_fixture="$tmp_dir/unrelated"
create_fixture "$unrelated_fixture"
printf '\n// unrelated local edit\n' >> "$unrelated_fixture/mlx/backend/metal/device.cpp"
run_migration "$unrelated_fixture"
tail -n 1 "$unrelated_fixture/mlx/backend/metal/device.cpp" |
  grep -Fxq '// unrelated local edit' || fail "unrelated edit was not preserved"
assert_applied "$unrelated_fixture" "$current_sdk"
assert_absent "$unrelated_fixture" "$obsolete_swift"
assert_applied "$unrelated_fixture" "$addrspace"
echo "PASS: unrelated edit preserved"

conflict_fixture="$tmp_dir/conflict"
create_fixture "$conflict_fixture"
git -C "$conflict_fixture" apply "$obsolete_swift"
perl -pi -e 's/xcrun -sdk macosx metal/xcrun -sdk custom metal/g' \
  "$conflict_fixture/mlx/backend/metal/kernels/CMakeLists.txt"
git -C "$conflict_fixture" diff --binary > "$tmp_dir/conflict-before.diff"
if "$apply_script" "$conflict_fixture" \
    --remove "$obsolete_swift" \
    --apply "$addrspace" \
    --migrate "$old_sdk" "$current_sdk"; then
  fail "conflicting SDK edit unexpectedly succeeded"
fi
git -C "$conflict_fixture" diff --binary > "$tmp_dir/conflict-after.diff"
cmp -s "$tmp_dir/conflict-before.diff" "$tmp_dir/conflict-after.diff" ||
  fail "failed migration did not restore the original working tree"
echo "PASS: conflict fails and transaction rolls back without data loss"

echo "All MLX patch migration tests passed."
