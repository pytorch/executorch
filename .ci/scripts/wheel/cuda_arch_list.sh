#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GPU architectures to compile device code for, chosen per release row rather than detected
# from the build machine.
#
# Without this, the build compiles for whichever GPU the builder happens to have. The wheel
# then installs on every machine the row claims and fails when a model runs on a different
# generation. Detection is the right default for a local build and the wrong one for a
# published artifact.
#
# The value is published as TORCH_CUDA_ARCH_LIST rather than CMAKE_CUDA_ARCHITECTURES.
# PyTorch's own CMake explicitly rejects the latter and overrides it with OFF, so setting it
# alone silently reduces the build to a single detected architecture.

# Data center and desktop parts on the CUDA 13 trains: Ampere, Hopper, Blackwell data center,
# and Blackwell desktop.
_cuda_arch_x86_64_cu130="8.0 9.0 10.0 12.0"
_cuda_arch_x86_64_cu132="${_cuda_arch_x86_64_cu130}"

# Server-class ARM plus the Jetson modules whose CUDA train matches: Hopper for Grace-Hopper,
# Blackwell data center for GB200, and Thor.
_cuda_arch_aarch64_cu130="9.0 10.0 11.0"
_cuda_arch_aarch64_cu132="${_cuda_arch_aarch64_cu130}"

# The older CUDA train, where Orin is the target.
_cuda_arch_aarch64_cu126="8.7"
_cuda_arch_x86_64_cu126="8.0 9.0"

# The architectures for the current row, space separated in the dotted form PyTorch expects.
# Empty when the row is unknown, which leaves the build detecting as before.
executorch_cuda_arch_list() {
  local machine
  machine="$(uname -m)"
  # The wheel build exports the row's CUDA train as CU_VERSION. DESIRED_CUDA is the name of
  # the matrix field, not of the variable, so reading only that one leaves every row falling
  # back to detecting the builder's GPU.
  local train="${CU_VERSION:-${DESIRED_CUDA:-}}"
  if [ -z "${train}" ]; then
    return 0
  fi
  # The value arrives as cu130; some callers pass 13.0 instead.
  train="${train#cu}"
  train="${train//./}"

  case "${machine}" in
    aarch64 | arm64)
      case "${train}" in
        126) printf '%s' "${_cuda_arch_aarch64_cu126}" ;;
        130) printf '%s' "${_cuda_arch_aarch64_cu130}" ;;
        132) printf '%s' "${_cuda_arch_aarch64_cu132}" ;;
      esac
      ;;
    x86_64)
      case "${train}" in
        126) printf '%s' "${_cuda_arch_x86_64_cu126}" ;;
        130) printf '%s' "${_cuda_arch_x86_64_cu130}" ;;
        132) printf '%s' "${_cuda_arch_x86_64_cu132}" ;;
      esac
      ;;
  esac
}

# The same architectures in CMake's own form, for targets outside PyTorch's CMake.
executorch_cuda_cmake_arch_list() {
  local dotted
  dotted="$(executorch_cuda_arch_list)"
  if [ -z "${dotted}" ]; then
    return 0
  fi
  local out="" entry
  for entry in ${dotted}; do
    entry="${entry//./}"
    out="${out:+${out};}${entry}-real"
  done
  printf '%s' "${out}"
}
