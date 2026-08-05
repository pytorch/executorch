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

# The older CUDA train on generic ARM. These are the server parts that train supports, matching
# what the x86_64 row of the same train claims. Jetson is deliberately not here: a Jetson-only
# architecture such as Orin's 8.7 in a generic manylinux wheel would advertise a device the row
# cannot otherwise serve, since a Jetson also needs the pinned CUDA, TensorRT and PyTorch from its
# own software release rather than the ones a generic wheel resolves.
_cuda_arch_aarch64_cu126="8.0 9.0"
_cuda_arch_x86_64_cu126="8.0 9.0"

# A CUDA train with no architecture list would otherwise leave the build detecting the
# builder's GPU, which is the failure this file exists to prevent. Adding a train to the release
# matrix without adding its architectures should fail loudly.
_executorch_unknown_train() {
  echo "cuda_arch_list.sh: no GPU architecture list for CUDA train '$1' on $(uname -m)." >&2
  echo "Add one before building this row, or the wheel ships device code for one GPU only." >&2
  return 64
}

# The architectures for the current row, space separated in the dotted form PyTorch expects.
# Empty when the row is unknown, which leaves the build detecting as before.
executorch_cuda_arch_list() {
  local machine
  machine="$(uname -m)"
  # The wheel build exports the row's CUDA train as CU_VERSION. DESIRED_CUDA is the name of
  # the matrix field, not of the variable, so reading only that one leaves every row falling
  # back to detecting the builder's GPU.
  local train="${CU_VERSION:-${DESIRED_CUDA:-}}"
  # A CPU row names no CUDA train and needs no architectures, so it is not an error.
  case "${train}" in
    "" | cpu | CPU | none | NONE) return 0 ;;
  esac
  # The value arrives as cu130; some callers pass 13.0 instead.
  train="${train#cu}"
  train="${train//./}"

  case "${machine}" in
    aarch64 | arm64)
      case "${train}" in
        126) printf '%s' "${_cuda_arch_aarch64_cu126}" ;;
        130) printf '%s' "${_cuda_arch_aarch64_cu130}" ;;
        132) printf '%s' "${_cuda_arch_aarch64_cu132}" ;;
        *) _executorch_unknown_train "${train}" ;;
      esac
      ;;
    x86_64)
      case "${train}" in
        126) printf '%s' "${_cuda_arch_x86_64_cu126}" ;;
        130) printf '%s' "${_cuda_arch_x86_64_cu130}" ;;
        132) printf '%s' "${_cuda_arch_x86_64_cu132}" ;;
        *) _executorch_unknown_train "${train}" ;;
      esac
      ;;
    *) _executorch_unknown_train "${train}" ;;
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
