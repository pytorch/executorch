#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Vendor Dawn (Tint) + SwiftShader for the WebGPU backend CI WITHOUT hosting a
# private prebuilt:
#   * Dawn  : Google's official nightly prebuilt, downloaded directly from
#             github.com/google/dawn/releases (pinned tag+rev+sha256) -- the same
#             "fetch a pinned upstream prebuilt" pattern used for other CI deps.
#   * SwiftShader : built from source at a pinned rev compatible with the Dawn
#             above (the ossci prebuilt is from 2020, too old for current Dawn). No S3.
# Dawn (Chrome's WebGPU impl; its WGSL compiler Tint is the spec reference) on
# SwiftShader gives a headless, deterministic, spec-faithful CLI backend.
#
# Exports Dawn_DIR / VK_ICD_FILENAMES / LD_LIBRARY_PATH for the cmake build+run.
# Local/rig override: set DAWN_PREBUILT_DIR=<dir containing lib64/cmake/Dawn> to
# skip the Dawn download.
set -ex

# --- pinned versions (bump rev+sha together when upgrading Dawn) --------------
DAWN_TAG="${DAWN_TAG:-v20260423.175430}"
DAWN_REV="${DAWN_REV:-31e25af254ab572c77054edec4946d2244e184dd}"
DAWN_SHA256="${DAWN_SHA256:-ac76fac090162dc1ecea5ed0f28a557bb8f49efc47faab01886105ace82b7b64}"
# SwiftShader rev verified compatible with DAWN_REV (the old ossci prebuilt is
# from 2020 and is incompatible with current Dawn -> no adapter / zero compute).
SWIFTSHADER_REV="${SWIFTSHADER_REV:-9898204d91d6a60b6a08ad74fe4ac52a6913111b}"

_dawn_dir="${DAWN_PREBUILT_DIR:-/tmp/dawn-ci}"
_ss_dir=/tmp/swiftshader

# --- toolchain prereqs --------------------------------------------------------
# Dawn dlopens the system Vulkan loader at runtime (libvulkan1). And the
# ubuntu-latest prebuilt is built with a bleeding-edge GCC: it references
# libstdc++ symbols newer than ubuntu-22.04's default (e.g. _M_replace_cold,
# GCC 13+), so the static .a won't link against the stock runtime. Pull a current
# libstdc++ from the ubuntu-toolchain-r PPA when the symbol floor isn't met. All
# of this is scoped to the WebGPU CI job; newer libstdc++ is backward-compatible.
if command -v apt-get >/dev/null 2>&1; then
  _SUDO=""; command -v sudo >/dev/null 2>&1 && _SUDO="sudo"
  ${_SUDO} apt-get update -y || true
  ${_SUDO} apt-get install -y libvulkan1 software-properties-common || true
  if ! strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null \
      | grep -q "GLIBCXX_3.4.32"; then
    ${_SUDO} add-apt-repository -y ppa:ubuntu-toolchain-r/test || true
    ${_SUDO} apt-get update -y || true
    ${_SUDO} apt-get install -y libstdc++6 || true  # newest GCC runtime
  fi
fi

# The native binaries / pybind lib run INSIDE the CI conda env, whose libstdc++
# predates GLIBCXX_3.4.32 (the Dawn prebuilt's floor) -- the same wall ssjia hit
# for the vulkan op tests. Upgrade the conda runtime libstdc++ so the loaded
# libstdc++.so.6 (conda's, not the system one) satisfies Dawn at run time.
if command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge "libstdcxx-ng>=14" || true
fi

# --- Dawn: official prebuilt from GitHub (no S3) ------------------------------
mkdir -p "${_dawn_dir}"
if [[ ! -d "${_dawn_dir}/lib64/cmake/Dawn" ]]; then
  _dawn_tar="/tmp/Dawn-${DAWN_REV}-ubuntu-latest-Release.tar.gz"
  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "${_dawn_tar}" \
    "https://github.com/google/dawn/releases/download/${DAWN_TAG}/Dawn-${DAWN_REV}-ubuntu-latest-Release.tar.gz"
  echo "${DAWN_SHA256}  ${_dawn_tar}" | sha256sum -c -
  # archive top dir is Dawn-<rev>-ubuntu-latest-Release/{lib64,include,bin}
  tar -C "${_dawn_dir}" --strip-components=1 -xzf "${_dawn_tar}"
fi

# --- SwiftShader: build from source at a pinned rev (no S3) -------------------
# The old ossci prebuilt (swiftshader-abe07b943, 2020) is incompatible with the
# current Dawn; build a matching modern SwiftShader instead. Self-contained
# cmake build (vendored LLVM); the ICD lands under build/<OS>/.
if [[ ! -d "${_ss_dir}/build" ]]; then
  if [[ ! -d "${_ss_dir}/.git" ]]; then
    git clone https://github.com/google/swiftshader "${_ss_dir}"
  fi
  git -C "${_ss_dir}" checkout "${SWIFTSHADER_REV}"
  # vk_swiftshader's deps are vendored in-tree; tolerate unreachable
  # disabled-feature submodules (angle, test-only) failing to fetch.
  git -C "${_ss_dir}" submodule update --init --recursive || true
  cmake -S "${_ss_dir}" -B "${_ss_dir}/build" -DCMAKE_BUILD_TYPE=Release \
    -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_BUILD_PVR=OFF \
    -DSWIFTSHADER_BUILD_BENCHMARKS=OFF
  cmake --build "${_ss_dir}/build" --parallel "$(nproc)" --target vk_swiftshader
fi
_ss_icd="$(find "${_ss_dir}/build" -name vk_swiftshader_icd.json 2>/dev/null | head -1)"
[[ -n "${_ss_icd}" ]] || { echo "ERROR: SwiftShader ICD not found after build" >&2; exit 1; }

_ss_libdir="$(dirname "${_ss_icd}")"
export Dawn_DIR="${_dawn_dir}/lib64/cmake/Dawn"
export VK_ICD_FILENAMES="${_ss_icd}"
export LD_LIBRARY_PATH="${_ss_libdir}:${LD_LIBRARY_PATH:-}"
export WEBGPU_USING_SWIFTSHADER=1
