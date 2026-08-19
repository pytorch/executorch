#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Verifies that the Android JNI shared library exports the curated
# EXECUTORCH_1_0 ABI defined in extension/android/jni/version_script.txt.
#
# External backend/kernel .so files rely on these symbols being visible in the
# core .so so that ELF symbol interposition gives them access to the
# process-wide backend/kernel registries at load time. If any of these symbols
# go missing (e.g. the version script is accidentally narrowed or stops being
# applied), split backends would silently register into a private registry and
# delegation would fail with "backend not found" at runtime. Fail the build
# instead.
#
# Usage: check_exported_symbols.sh <path-to-libexecutorch.so> [path-to-llvm-nm]

set -euo pipefail

SO_PATH="${1:?usage: check_exported_symbols.sh <libexecutorch.so> [llvm-nm]}"
NM="${2:-llvm-nm}"

# Demangled symbol fragments that must be exported by the core .so.
REQUIRED_SYMBOLS=(
  # Backend registration (runtime/backend/interface.h)
  "executorch::runtime::register_backend"
  "executorch::runtime::get_backend_class"
  # BackendInterface RTTI/vtable; external backends subclass it and the key
  # function (the destructor) is emitted in the core .so.
  "typeinfo for executorch::runtime::BackendInterface"
  "vtable for executorch::runtime::BackendInterface"
  # Kernel registration (runtime/kernel/operator_registry.h)
  "executorch::runtime::register_kernels"
  # exec_aten data model used by backend implementations (the portable Tensor
  # is executorch::runtime::etensor::Tensor; executorch::aten::Tensor is a
  # using-alias, so symbols carry the real namespace).
  "executorch::runtime::etensor::Tensor"
  "executorch::runtime::EValue"
  # PAL hooks so backend .so files share process-wide logging
  "et_pal_emit_log_message"
  # JNI entry points (pre-existing export contract)
  "JNI_OnLoad"
)

EXPORTED="$("${NM}" -D --defined-only -C "${SO_PATH}")"

missing=0
for sym in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -qF -- "${sym}" <<<"${EXPORTED}"; then
    echo "ERROR: ${SO_PATH} does not export required ABI symbol: ${sym}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "The EXECUTORCH_1_0 ABI surface is incomplete." >&2
  echo "See extension/android/jni/version_script.txt and" >&2
  echo "https://github.com/pytorch/executorch/issues/10457 (milestone 1)." >&2
  exit 1
fi

echo "OK: ${SO_PATH} exports the EXECUTORCH_1_0 ABI surface."
