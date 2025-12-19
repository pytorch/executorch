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
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  STANDALONE_HOST_DEVICE explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace executorch::backends::aoti::slim::c10
