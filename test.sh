#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -eu

function _build_targets() {
  local platform="${1}"
  local flags="${2:-}"

  echo -e "\n🟡 Building for ${platform}"
  rm -rf cmake-out
  local start_time=$(date +%s)
  cmake -DEXECUTORCH_H12025_BUILD_MIGRATION=ON ${flags} -S . -B cmake-out
  cmake --build cmake-out --parallel $(sysctl -n hw.ncpu) --target help all
  local end_time=$(date +%s)
  echo "✅ ${platform} in $((end_time - start_time))s"
}

_build_targets "host"
_build_targets "iOS" "-DPLATFORM_TARGET_OS=ios"
