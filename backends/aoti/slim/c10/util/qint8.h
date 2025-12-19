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
 * This is the data type for quantized Tensors. Right now we only have
 * qint8 which is for 8 bit Tensors, and qint32 for 32 bit int Tensors,
 * we might have 4 bit, 2 bit or 1 bit data types in the future.
 */
struct alignas(1) qint8 {
  using underlying = int8_t;
  int8_t val_;
  qint8() = default;
  STANDALONE_HOST_DEVICE explicit qint8(int8_t val) : val_(val) {}
};

} // namespace executorch::backends::aoti::slim::c10
