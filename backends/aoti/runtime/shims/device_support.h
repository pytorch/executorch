/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/log.h>
#include <string>

namespace executorch {
namespace backends {
namespace aoti {

/**
 * Returns the appropriate device type based on compile-time defines.
 * 1: CUDA
 * 2: Metal/MPS
 * 0: CPU (fallback)
 */
inline int32_t get_aoti_backend_device_type() {
#if defined(AOTI_CUDA)
  return 1; // CUDA
#elif defined(AOTI_METAL)
  return 2; // Metal/MPS
#else
  // Neither supported, use CPU (0)
  return 0;
#endif
}

/**
 * Returns the appropriate device string to use with AOTInductor
 * "cuda": CUDA device
 * "mps": Metal device
 * "cpu": CPU fallback
 */
inline const char* get_aoti_backend_device_string() {
#if defined(AOTI_CUDA)
  ET_LOG(Info, "Using CUDA device");
  return "cuda";
#elif defined(AOTI_METAL)
  ET_LOG(Info, "Using Metal/MPS device");
  return "mps";
#else
  // Neither supported, use CPU
  ET_LOG(Warning, "No GPU support available, falling back to CPU");
  return "cpu";
#endif
}

} // namespace aoti
} // namespace backends
} // namespace executorch
