#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GPU architectures to compile device code for, chosen per release row rather than detected
# from the build machine.
#
# Without this, CMake compiles for whichever GPU the builder happens to have. The wheel then
# installs on every machine the row claims and fails when a model runs on a different GPU
# generation. Detection is the right default for a local build and the wrong one for a
# published artifact.
#
# CUDA_ARCH_LIST_<major><minor> names the architectures for that CUDA train. DESIRED_CUDA is
# supplied by the wheel build as cu126, cu130, and so on.

# Data center and desktop parts on the CUDA 13 trains: Ampere, Hopper, Blackwell data center,
# and Blackwell desktop.
_cuda_arch_x86_64_cu130="80-real;90-real;100-real;120-real"
_cuda_arch_x86_64_cu132="${_cuda_arch_x86_64_cu130}"

# Server-class ARM plus the Jetson modules whose CUDA train matches: Hopper for Grace-Hopper,
# Blackwell data center for GB200, and Thor.
_cuda_arch_aarch64_cu130="90-real;100-real;110-real"
_cuda_arch_aarch64_cu132="${_cuda_arch_aarch64_cu130}"

# The older CUDA train, where Orin is the target.
_cuda_arch_aarch64_cu126="87-real"
_cuda_arch_x86_64_cu126="80-real;90-real"

executorch_cuda_arch_list() {
  local machine
  machine="$(uname -m)"
  local train="${DESIRED_CUDA:-}"
  if [ -z "${train}" ]; then
    return 0
  fi
  # DESIRED_CUDA arrives as cu130; some callers pass 13.0 instead.
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
