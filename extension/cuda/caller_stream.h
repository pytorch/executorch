/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <optional>
#include <type_traits>

#include <executorch/extension/cuda/export.h>

namespace executorch::extension::cuda {

/**
 * The CUDA stream selected by the innermost CallerStreamGuard active on this
 * thread, or std::nullopt if none is active.
 *
 * This reports only a stream the caller explicitly selected, so a backend can
 * honor that choice or fall back to its own default. It is backend-neutral: any
 * CUDA backend (e.g. the CUDA/AOTI delegate and the TensorRT delegate) can
 * consult it, so a single caller-provided stream -- including a CUDA
 * green-context stream -- can drive several delegates in one program.
 */
EXECUTORCH_EXTENSION_CUDA_API std::optional<cudaStream_t> getCallerStream();

/**
 * Scopes, for the calling thread, the CUDA stream a backend should run on, and
 * restores the previous selection on destruction. Scope it on the thread that
 * runs the call; the selection is one value per thread.
 *
 * A stream created with cuGreenCtxStreamCreate confines work to that green
 * context's SM partition; the confinement rides the stream, so the green
 * context need not be made current. The caller owns the stream for the guard's
 * lifetime.
 */
class EXECUTORCH_EXTENSION_CUDA_API CallerStreamGuard {
 public:
  explicit CallerStreamGuard(cudaStream_t stream);
  ~CallerStreamGuard();
  CallerStreamGuard(const CallerStreamGuard&) = delete;
  CallerStreamGuard& operator=(const CallerStreamGuard&) = delete;
  CallerStreamGuard(CallerStreamGuard&&) = delete;
  CallerStreamGuard& operator=(CallerStreamGuard&&) = delete;

 private:
  std::optional<cudaStream_t> previous_;
};

// std::optional<cudaStream_t> is trivially copyable (asserted below), so it
// crosses the shared-library boundary unaffected by the libstdc++ CXX11 ABI,
// which only changes the layout of types like std::string and std::list.
static_assert(std::is_trivially_copyable_v<std::optional<cudaStream_t>>);

} // namespace executorch::extension::cuda
