#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GPU architectures to compile device code for, chosen per release row rather than detected from
# the build machine.
#
# Without this nothing selects the architectures, so nvcc falls back to its own default and the
# wheel carries device code for that one architecture regardless of the builder's GPU. Measured:
# with the architecture list unset the compile line has no gencode flags at all. The wheel then
# installs on every machine the row claims and fails when a model runs on a different generation,
# with an error that looks like a model problem rather than a packaging one. Detection is the right
# default for a local build and the wrong one for a published artifact.
#
# The value is published as TORCH_CUDA_ARCH_LIST rather than CMAKE_CUDA_ARCHITECTURES, because
# PyTorch's CMake rejects the latter and overrides it, so setting only that reduces the build to a
# single detected architecture.

# The architectures each row serves. Two rules decide the list, and they pull in opposite directions.
#
# The upper end follows the published PyTorch build for that train, read from its own library rather than
# chosen by reasoning about which GPUs matter. A delegate is only useful where torch already runs, and an
# architecture torch supports but this wheel omits produces a wheel that installs and then fails at the
# first kernel launch. Two omissions found that way were the GPU on the runner that tests these wheels,
# and a common desktop card.
#
# The lower end does NOT follow torch. It stops at 8.0 even though torch reaches further down, because one
# source here compiles an integer matrix-multiply path only at 8.0 and above. Below that a user gets a
# delegate that loads, runs most models, and fails on one needing that operator, which is worse than a row
# that never claimed the device. So these lists are narrower than torch at the bottom on purpose.
_cuda_arch_x86_64_cu130="8.0 8.6 8.9 9.0 10.0 12.0"
_cuda_arch_x86_64_cu132="${_cuda_arch_x86_64_cu130}"

# The architectures the published aarch64 PyTorch CUDA build covers, read from its own library on an ARM
# machine, for the same reason as the x86_64 rows above. Includes the ARM module whose train matches.
_cuda_arch_aarch64_cu130="8.0 9.0 10.0 11.0 12.0"
_cuda_arch_aarch64_cu132="${_cuda_arch_aarch64_cu130}"

# The older CUDA train.
#
# The two architectures do not carry identical lists, because each covers what the published PyTorch
# build for that architecture covers, and those differ. Matching them to each other instead would mean
# advertising a GPU on one architecture that PyTorch cannot serve there.
#
# The smaller embedded modules are deliberately absent, with one exception. An embedded-only
# architecture in a generic wheel would advertise a device the row cannot otherwise serve, since
# those devices also need the CUDA, TensorRT and PyTorch pinned by their own software release
# rather than the ones a generic wheel resolves.
#
# 8.7 is that exception. This is the only row whose CUDA major matches what that module's software
# release ships, and the wheel declares no PyTorch, so the user supplies the build that carries
# their architecture. Omitting it does not protect them from a bad pairing, it only removes the
# device code they need.
#
# The floor is 8.0 rather than the oldest architecture PyTorch still carries. One of these sources compiles
# an integer matrix-multiply path only at 8.0 and newer, so an older architecture would get a delegate that
# loads, runs most models, and fails on one that needs that operator. Claiming hardware the delegate only
# partly serves is the same problem the embedded modules have, so the row leaves it out for the same reason.
_cuda_arch_x86_64_cu126="8.0 8.6 8.9 9.0"
_cuda_arch_aarch64_cu126="8.0 8.7 9.0"

# A CUDA train with no architecture list would leave the build detecting the builder's GPU, which is
# the failure this file exists to prevent. Adding a train to the release matrix without adding its
# architectures should fail loudly rather than silently produce a single-GPU wheel.
_executorch_unknown_train() {
  echo "cuda_arch_list.sh: no GPU architecture list for CUDA train '$1' on $(uname -m)." >&2
  echo "Add one before building this row, or the wheel ships device code for one GPU only." >&2
  return 64
}

# The architectures for the current row, space separated in the dotted form PyTorch expects.
executorch_cuda_arch_list() {
  local machine
  machine="$(uname -m)"
  # The wheel build exports the row's CUDA train as CU_VERSION. DESIRED_CUDA is the name of the
  # matrix field rather than of the variable, so reading only that leaves every row falling back to
  # detecting the builder's GPU.
  local train="${CU_VERSION:-${DESIRED_CUDA:-}}"
  # A CPU row names no CUDA train and needs no architectures, so it is not an error.
  #
  # A CUDA row always names one, so an empty value there means the row lost it. Treating that as a CPU
  # row let the build fall back to detecting the builder's GPU, which produces a wheel carrying device
  # code for whatever machine happened to build it while every check still reports green.
  case "${train}" in
    "" | cpu | CPU | none | NONE)
      if [ "${EXECUTORCH_BUILD_CUDA:-}" = "1" ]; then
        echo "this is a CUDA build but the row's CUDA version is '${train}', which names no CUDA" >&2
        echo "train. Refusing to detect the builder GPU instead." >&2
        return 65
      fi
      return 0
      ;;
  esac
  # The value arrives as cu130, while some callers pass 13.0 instead.
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

# The architecture list for a row. Every entry already carries the portable form that lets a newer
# GPU compile at load time: an unsuffixed architecture asks the compiler for both the compiled and
# the portable form, measured as code=[compute_120,sm_120] for a bare "120". Nothing extra is needed
# for forward compatibility.
executorch_cuda_arch_list_with_ptx() {
  local dotted
  # Propagate a failed lookup rather than reporting an empty list, since a caller cannot tell an
  # unknown row from a CPU row and the unknown one must not pass silently.
  dotted="$(executorch_cuda_arch_list)" || return $?
  [ -n "${dotted}" ] || return 0
  printf '%s' "${dotted}"
}
