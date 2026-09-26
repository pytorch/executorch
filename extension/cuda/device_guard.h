/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/cuda/export.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::extension::cuda {

/**
 * Selects a CUDA device for its own lifetime and puts the caller's device back
 * on destruction.
 *
 * The current device belongs to whoever called in, so a backend that needs its
 * own device borrows it rather than keeping it. Kernel launches follow the
 * stream they are given, but allocation follows the current device, so a
 * backend that allocates has to select one even when the caller chose the
 * stream.
 *
 * Backend-neutral for the reason CallerStreamGuard is: one program may run
 * several backends whose engines sit on different devices, so each has to leave
 * the device as it found it, or the next one inherits a selection it never
 * made.
 *
 * No CUDA header is reached for here. Nothing in this interface names a CUDA
 * type, so a consumer can include this against the wheel alone, the same as
 * every other header it ships.
 */
class EXECUTORCH_EXTENSION_CUDA_API CUDAGuard {
 public:
  /**
   * Borrows @p device_index for the returned guard's lifetime. The result
   * carries the failure instead of the guard, so a caller cannot use a guard
   * that never selected anything.
   */
  static ::executorch::runtime::Result<CUDAGuard> create(int device_index);

  /** Restores the device that was current when this guard selected its own. */
  ~CUDAGuard();

  /**
   * Takes over the borrow. The moved-from guard is left looking already
   * restored, so only one of the two puts the device back.
   */
  CUDAGuard(CUDAGuard&& other) noexcept;
  CUDAGuard& operator=(CUDAGuard&&) = delete;
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  /** Selects @p device_index, recording the one it replaced. */
  ::executorch::runtime::Error set_index(int device_index);

  /** The device that was current before this guard selected its own. */
  int original_device() const {
    return original_device_index_;
  }

  /** The device this guard selected. */
  int current_device() const {
    return current_device_index_;
  }

 private:
  CUDAGuard() = default;

  int original_device_index_{-1};
  int current_device_index_{-1};
};

} // namespace executorch::extension::cuda
