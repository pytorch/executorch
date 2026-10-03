/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch::backends::cuda {

// Shared identifiers for the CUDA backend. Included by the backend itself (to
// register) and by callers/runners (to route backend options and to ask
// CacheFactory for a builder), keeping the string literals in one place.

// Backend id under which the CUDA backend registers (see cuda_backend.cpp).
inline constexpr char kCudaBackendId[] = "CudaBackend";

} // namespace executorch::backends::cuda
