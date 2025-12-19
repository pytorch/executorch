/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cstdint>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>

namespace executorch::backends::aoti::slim::c10 {

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  STANDALONE_HOST_DEVICE explicit qint32(int32_t val) : val_(val) {}
};

} // namespace executorch::backends::aoti::slim::c10
