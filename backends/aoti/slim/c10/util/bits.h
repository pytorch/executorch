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
 * bits1x8 is an uninterpreted dtype of a tensor with 1 bit (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits1x8 {
  using underlying = uint8_t;
  uint8_t val_;
  bits1x8() = default;
  STANDALONE_HOST_DEVICE explicit bits1x8(uint8_t val) : val_(val) {}
};

/**
 * bits2x4 is an uninterpreted dtype of a tensor with 2 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits2x4 {
  using underlying = uint8_t;
  uint8_t val_;
  bits2x4() = default;
  STANDALONE_HOST_DEVICE explicit bits2x4(uint8_t val) : val_(val) {}
};

/**
 * bits4x2 is an uninterpreted dtype of a tensor with 4 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits4x2 {
  using underlying = uint8_t;
  uint8_t val_;
  bits4x2() = default;
  STANDALONE_HOST_DEVICE explicit bits4x2(uint8_t val) : val_(val) {}
};

/**
 * bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
 * semantics defined.
 */
struct alignas(1) bits8 {
  uint8_t val_;
  bits8() = default;
  STANDALONE_HOST_DEVICE explicit bits8(uint8_t val) : val_(val) {}
};

/**
 * bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(2) bits16 {
  uint16_t val_;
  bits16() = default;
  STANDALONE_HOST_DEVICE explicit bits16(uint16_t val) : val_(val) {}
};

} // namespace executorch::backends::aoti::slim::c10
